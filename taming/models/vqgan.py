import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config,save_img


from taming.modules.diffusionmodules.model import Encoder, Decoder,CTCAuxWrapper
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer
import torch.distributed  as dst
import numpy as np
import os
from PIL import Image



class VQModelCTCAux(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 pretrain_ocr = False,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.ctc_head  = CTCAuxWrapper()
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_counter = 0
        self.pretrain_ocr = pretrain_ocr
        # ##freeze_all
        # self.encoder.requires_grad_(False)
        # self.decoder.requires_grad_(False)
        # self.quantize.requires_grad_(False)
        # self.quant_conv.requires_grad_(False)
        # self.post_quant_conv.requires_grad_(False)
        # self.encoder.eval()
        # self.decoder.eval()
        # self.quantize.eval()
        # self.quant_conv.eval()
        # self.post_quant_conv.eval()


    # def unfreeze(self):
    #     ##freeze_all
    #     self.encoder.train()
    #     self.decoder.train()
    #     self.quantize.train()
    #     self.quant_conv.train()
    #     self.post_quant_conv.train()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input,batch_1):
        quant, diff, _ = self.encode(input)
        ctc_loss,title_pair,legened_pair =self.ctc_head(quant,batch_1)
        dec = self.decode(quant)
        return dec, diff,ctc_loss

    def get_input(self, batch, k):
        x = batch[0][k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()


    def training_step(self, batch, batch_idx, optimizer_idx):
        self.batch_counter+=1
        # if self.batch_counter == 100:
        #     self.reconfigure_optimizers()
        #     print("reconfigure_optimizers!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     dst.barrier()
        x = self.get_input(batch, self.image_key)
        xrec, qloss,ctc_loss = self(x,batch[1])
        if True:#self.batch_counter<100:
            if optimizer_idx == 0:
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), metadata = batch[1],split="train")
                aeloss = aeloss+0.1*ctc_loss
                log_dict_ae["ctc_loss"]=ctc_loss
                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return aeloss

            if optimizer_idx == 1:
                # discriminator
                discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                    last_layer=self.get_last_layer(), split="train")
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return discloss
        # else:
        #     aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
        #                                     last_layer=self.get_last_layer(), split="train")
        #     aeloss =  0.1 * ctc_loss
        #     log_dict_ae["ctc_loss"] = ctc_loss
        #     self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        #     self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        #     return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        #title,legend_location,legend_of_plots,colors = self.get_ctclabels(batch)
        xrec, qloss,ctc_loss = self(x,batch[1])
        aeloss, log_dict_ae  = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), metadata = batch[1], split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        aeloss = aeloss + 0.1 * ctc_loss
        log_dict_ae["ctc_loss"] = ctc_loss
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return self.log_dict

    def test_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss,ctc_loss = self(x,batch[1])

        _, log_dict = self.loss(qloss, x, xrec, 0, self.global_step,
                                last_layer=self.get_last_layer(), metadata = batch[1], split="test")
        log_dict["ctc_loss"] = ctc_loss
        # Compute and Store reconstructions
        im_rec = xrec.detach().cpu()
        im_rec = torch.clamp(im_rec, -1., 1.)
        im_rec = (im_rec + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        im_rec = im_rec.transpose(1, 2).transpose(2, 3)
        im_rec = im_rec.numpy()
        im_rec = (im_rec * 255).astype(np.uint8)

        for k in range(im_rec.shape[0]):
            filename = f"reconstruction_batch_{batch_idx}_id_{k}.png"
            path = os.path.join(self.trainer.logdir, 'evaluation', filename)
            im = im_rec[k]
            Image.fromarray(im).save(path)

        # Compute LPIPS
        LPIPS = log_dict["test/p_loss"]
        try:
            OCR_loss = log_dict["test/p_ocr_loss"]
        except:
            OCR_loss = 0.0

        output = dict({
            'LPIPS': LPIPS,
            'OCR_loss': OCR_loss
        })
        self.log_dict(output)
        return output

    def reconfigure_optimizers(self):#configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam([{"params":list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),"lr":0},
                                  {"params":list(self.ctc_head.parameters())}],
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=0, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def configure_optimizers(self):
        lr = self.learning_rate
        if self.pretrain_ocr:
            opt_ae = torch.optim.Adam([{"params": list(self.encoder.parameters()) +
                                                  list(self.decoder.parameters()) +
                                                  list(self.quantize.parameters()) +
                                                  list(self.quant_conv.parameters()) +
                                                  list(self.post_quant_conv.parameters()), "lr": 0},
                                       {"params": list(self.ctc_head.parameters())}],
                                      lr=lr, betas=(0.5, 0.9))
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=0, betas=(0.5, 0.9))
        else:
            opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                      list(self.decoder.parameters()) +
                                      list(self.quantize.parameters()) +
                                      list(self.quant_conv.parameters()) +
                                      list(self.post_quant_conv.parameters())+
                                      list(self.ctc_head.parameters()),
                                      lr=lr, betas=(0.5, 0.9))
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _,_ = self(x,batch[1])
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = torch.cat([x,xrec],dim=-1)
        #log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x







