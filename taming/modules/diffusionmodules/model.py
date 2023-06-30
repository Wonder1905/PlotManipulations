# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
#from torchaudio.models.decoder import ctc_decoder
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce
from torch import nn, einsum

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t


def l2norm(t):
    return F.normalize(t, dim = -1)
class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        scale = 8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.LayerNorm(dim)
        )

    def forward(self, x, latents, mask = None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # qk rmsnorm

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # similarities and masking

        sim = einsum('... i d, ... j d  -> ... i j', q, k) * self.scale

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class LayerNorm(nn.Module):
    def __init__(self, feats, stable = False, dim = -1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        if self.stable:
            x = x / x.amax(dim = dim, keepdim = True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = dim, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = dim, keepdim = True)

        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)
def FeedForward(dim, mult = 2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias = False)
    )
#ChanLayerNorm = partial(LayerNorm, dim = -3)
def masked_mean(t, *, dim, mask = None):
    if not exists(mask):
        return t.mean(dim = dim)

    denom = mask.sum(dim = dim, keepdim = True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim = dim) / denom.clamp(min = 1e-5)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_latents_mean_pooled = 4, # number of latents derived from mean pooled representation of the sequence
        max_seq_len = 512,
        ff_mult = 4
    ):
        super().__init__()
        self.convert_inputdim2dim = nn.Linear(input_dim,dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.to_latents_from_mean_pooled_seq = None

        if num_latents_mean_pooled > 0:
            self.to_latents_from_mean_pooled_seq = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange('b (n d) -> b n d', n = num_latents_mean_pooled)
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, mask = None):
        x = self.convert_inputdim2dim(x)
        n, device = x.shape[1], x.device
        pos_emb = self.pos_emb(torch.arange(n, device = device))

        x_with_pos = x + pos_emb

        latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])

        if exists(self.to_latents_from_mean_pooled_seq):
            meanpooled_seq = masked_mean(x, dim = 1, mask = torch.ones(x.shape[:2], device = x.device, dtype = torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim = -2)

        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask = mask) + latents
            latents = ff(latents) + latents

        return latents

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)  # Here we get -infs in level 1 block 2
            # for param in self.conv.parameters():
            #     print(param)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

class CrossAttnBlock(nn.Module):
    def __init__(self, in_channels,ctc_text=False):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)


        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        if not ctc_text:

            self.k = torch.nn.Linear(in_channels,
                                     in_channels)
            self.v = torch.nn.Linear(in_channels,
                                     in_channels)

        else:
            self.title2embdim = torch.nn.Conv1d(256,
                                                in_channels,
                                                kernel_size=5,
                                                stride=2,
                                                padding=2)
            self.leg2embdim = torch.nn.Conv1d(256,
                                              in_channels,
                                              kernel_size=5,
                                              stride=2,
                                              padding=2)
            self.gelu = nn.GELU()
            self.normctc = Normalize(in_channels)
            self.k = nn.Linear(68,in_channels)
            self.v = nn.Linear(68,in_channels)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
    def prepare_ctc_inputs(self,title_embs,legend_embs):
        title_embs = self.gelu(self.title2embdim(title_embs))
        legend_embs = self.gelu(self.leg2embdim(legend_embs))
        out = torch.cat([title_embs, legend_embs],dim=-1)
        out = self.normctc(out)
        return out
    def forward(self, x,text_emb=None,title_embs=None,legend_embs=None):
        if  title_embs is not None:
            gen_text_embs =self.prepare_ctc_inputs(title_embs,legend_embs)
        elif text_emb is not None:
            gen_text_embs = text_emb.to(self.k.weight.device)
            #gen_text_embs = rearrange(gen_text_embs,"1 d -> 1 1 d")
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(gen_text_embs).permute(0, 2, 1)
        v = self.v(gen_text_embs).permute(0, 2, 1)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k )  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

class CTCAuxHeads(nn.Module):
    ##Input   B,256,24,24
    ##Output  1.title ctc prediction,i.e:B,30.2legend title ctc i.e.: B,45, 15 for each possible pred
    def __init__(self, num_classes=2, image_size=384):
        super().__init__()
        spatial_input_size = image_size // 16
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1)
        self.ln1 = nn.LayerNorm((384, spatial_input_size // 2, spatial_input_size // 2))
        self.conv2 = nn.Conv2d(384, 256, kernel_size=3, stride=2, padding=1)
        self.ln2 = nn.LayerNorm((256, spatial_input_size // 4, spatial_input_size // 4))
        self.lin1 = nn.Linear((spatial_input_size // 4)**2, 64)
        self.ln_lin1 = nn.LayerNorm(64)
        self.title_ctc_head = nn.Linear(64, 35)
        self.ln_tit = nn.LayerNorm((256, 35))
        self.title_ctc_head1 = nn.Conv1d(256, num_classes, kernel_size=1, padding=0)
        self.legends_ctc_head = nn.Linear(64, 60)
        self.ln_lop = nn.LayerNorm((256, 60))
        self.legends_ctc_head1 = nn.Conv1d(256, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        B = x.shape[0]
        out = self.conv1(x)
        out = self.ln1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.ln2(out)
        out = self.gelu(out).view(B, 256, -1)
        out = self.lin1(out)
        out = self.gelu(self.ln_lin1(out))
        title_out = self.title_ctc_head(out)
        title_emb = title_out
        title_out = self.gelu(self.ln_tit(title_out))
        title_out = self.title_ctc_head1(title_out).transpose(2, 1)
        legends_out = self.legends_ctc_head(out)
        legends_emb = legends_out
        legends_out = self.gelu(self.ln_lop(legends_out))
        legends_out = self.legends_ctc_head1(legends_out).transpose(2, 1)
        return title_out, legends_out,title_emb,legends_emb

class ColorPredictionHeadEnc(nn.Module):
    ##Input   B,256,24,24
    ##Output  1.title ctc prediction,i.e:B,30.2legend title ctc i.e.: B,45, 15 for each possible pred
    def __init__(self, num_of_color_combinations=68):
        super().__init__()
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv2d(256, 768, kernel_size=3, stride=2, padding=1)
        self.ln1 = nn.LayerNorm((768, 4, 4))
        self.pool = nn.AdaptiveAvgPool2d(768)
        self.lin = nn.Linear(768, num_of_color_combinations)

    def forward(self, x):
        B = x.shape[0]
        out = self.conv1(x)
        out = self.ln1(out)
        out = self.gelu(out)
        out = self.pool(out)
        out = self.lin(out)
        return out

class CTCAuxWrapper(nn.Module):
    ##Input   B,256,24,24
    ##Output  1.title ctc prediction,i.e:B,30.2legend title ctc i.e.: B,45, 15 for each possible pred
    def __init__(self, num_classes=68, image_size=384):
        super().__init__()
        self.ctchead = CTCAuxHeads(num_classes=num_classes, image_size=image_size)

    def get_ctclabels(self, batch):
        x = batch
        title = x["title_tokenized"]
        title_str = x["title"]
        title_pad = x["title_tokenized_pad"]
        legend_location = x["legend_location"]
        legend_of_plots = x["lop_concat"]
        legend_of_plots_pads = x["lop_pads"]
        colors = []
        for line_idx in range(len(x["lines"])):
            line = x["lines"][line_idx]
            if "color" in line:
                colors.append(line["color"])
        return title, legend_location, legend_of_plots, colors, title_pad, legend_of_plots_pads,title_str

    # def add_tokenizer(self, data):
    #     self.tokenizer = data.tokenizer
    #     self.data_length_lop = data.length_lop
    #     self.loss = nn.CTCLoss(
    #         blank=self.tokenizer.encoder["<pad>"], zero_infinity=True)

    def get_loss(self, output, targets, pad):
        log_probs = F.log_softmax(output, dim=-1)
        non_mask_each_input = torch.LongTensor([len(b) for b in log_probs])
        log_probs = log_probs.permute(1, 0, 2)
        target_lengths = torch.Tensor([b.shape[0] - pad[j].item() for j, b in enumerate(
            targets)]).int()  # torch.Tensor(list(chain.from_iterable(x["labels"]))).int()
        #if torch.any(targets!=66):
        loss = self.loss(log_probs, targets, non_mask_each_input, target_lengths)
        #else:
        #    loss=None
        return loss

    def forward(self, x, titles_emb,titles_emb_msak,lop_concats_emb,lop_concats_emb_mask):
        title, legend_location, legend_of_plots, colors, title_pad, legend_of_plots_pads,title_str = self.get_ctclabels(batch_1)
        out_title, legends_out, title_emb, legends_emb = self.ctchead(x)
        ctc_loss = self.get_loss(out_title, title, title_pad)

        for m in range(len(colors)):
            lop_out = legends_out[:, m * self.data_length_lop:(m + 1) * self.data_length_lop, :]
            lop_target = legend_of_plots[:, m * self.data_length_lop:(m + 1) * self.data_length_lop]
            try:
                temp =  self.get_loss(lop_out, lop_target,legend_of_plots_pads[m])
                if temp<0:
                    print(title, temp)
                ctc_loss =ctc_loss +2*temp

            except:
                print(lop_out.shape, lop_target.shape,legend_of_plots_pads[m])
        out_title_char = self.tokenizer.batch_decode(torch.argmax(out_title, dim=-1), filter_word_delimiter_token=False)
        legends_outchar = self.tokenizer.batch_decode(torch.argmax(legends_out[0], dim=-1), filter_word_delimiter_token=False)


        return ctc_loss, (out_title_char, title), (legends_outchar, legend_of_plots[0]),out_title, legends_out,title_emb, legends_emb


class Model(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t=None):
        # assert x.shape[2] == x.shape[3] == self.resolution

        if self.use_timestep:
            # timestep embedding
            assert t is not None
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False,use_ctc_emb=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.use_ctc_emb = use_ctc_emb
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order
        self.cross_attention = CrossAttnBlock(512)
        # end
        #if self.use_ctc_emb:
        #    self.crossattn_iblock1 = CrossAttnBlock()
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
    #legends_emb=[B,256,35],title_emb=[B,256,60]
    def forward(self, z,text_emb=None,title_emb=None, legends_emb=None):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)
        # middle
        h = self.mid.block_1(h, temb)
        if title_emb is not None:
            h = self.cross_attention(h,title_emb = title_emb,legends_emb=legends_emb)
        elif text_emb is not None:
            h = self.cross_attention(h,text_emb=text_emb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        if title_emb is not None:
            h = self.cross_attention(h,title_emb=title_emb,legends_emb=legends_emb)
        elif text_emb is not None:
            h = self.cross_attention(h,text_emb=text_emb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)


        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DecoderCond(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False,use_ctc_emb=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.use_ctc_emb = use_ctc_emb
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(CrossAttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order
        self.cross_attention = CrossAttnBlock(512)
        # end
        #if self.use_ctc_emb:
        #    self.crossattn_iblock1 = CrossAttnBlock()
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
    #legends_emb=[B,256,35],title_emb=[B,256,60]
    def forward(self, z,text_emb=None,title_embs=None, legends_embs=None):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)
        # middle
        h = self.mid.block_1(h, temb)
        # if title_embs is not None:
        #     h = self.cross_attention(h,title_embs = title_embs,legends_embs=legends_embs)
        # elif text_emb is not None:
        #     h = self.cross_attention(h,text_emb=text_emb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # if title_emb is not None:
        #     h = self.cross_attention(h,title_embs=title_embs,legends_embs=legends_embs)
        # elif text_emb is not None:
        #     h = self.cross_attention(h,text_emb=text_emb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h,text_emb=text_emb)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h_before = nonlinearity(h)
        h = self.conv_out(h_before)
        return h,h_before


class DecoderCondAlsoOnTextPred(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False,use_ctc_emb=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.use_ctc_emb = use_ctc_emb
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(CrossAttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order
        self.cross_attention = CrossAttnBlock(512)
        # end
        #if self.use_ctc_emb:
        #    self.crossattn_iblock1 = CrossAttnBlock()
        self.norm_out = Normalize(block_in)


        self.combine_text_map = ResnetBlock(in_channels=256,
                                       out_channels=64,
                                       temb_channels=128,
                                       dropout=0.)
        #self.attn_combine_text_map = AttnBlock(64)
        self.combine_text_map2 = ResnetBlock(in_channels=64,
                                       out_channels=32,
                                       temb_channels=32,
                                       dropout=0.)
        self.conv_out = torch.nn.Conv2d(32,
                                        3,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
    #legends_emb=[B,256,35],title_emb=[B,256,60]
    def forward(self, z,text_map,text_emb=None,title_embs=None, legends_embs=None):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)
        # middle
        h = self.mid.block_1(h, temb)
        # if title_embs is not None:
        #     h = self.cross_attention(h,title_embs = title_embs,legends_embs=legends_embs)
        # elif text_emb is not None:
        #     h = self.cross_attention(h,text_emb=text_emb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # if title_emb is not None:
        #     h = self.cross_attention(h,title_embs=title_embs,legends_embs=legends_embs)
        # elif text_emb is not None:
        #     h = self.cross_attention(h,text_emb=text_emb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h,text_emb=text_emb)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.combine_text_map(torch.cat([h,text_map],dim=1),temb)
        #h = self.attn_combine_text_map(h)
        h = self.combine_text_map2(h,temb)
        h = self.conv_out(h)
        return h


class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution,
                 ch_mult=(2, 2), dropout=0.0):
        super().__init__()
        # upsampling
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # upsampling
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
