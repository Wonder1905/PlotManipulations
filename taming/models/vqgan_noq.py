import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config,save_img

from functools import partial, wraps
from taming.modules.diffusionmodules.model import Encoder, Decoder,CTCAuxWrapper,DecoderCond,PerceiverResampler,ColorPredictionHeadEnc,DecoderCondAlsoOnTextPred

import torch.distributed  as dst
import numpy as np
import os
from PIL import Image
from taming.modules.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME,t5_tokenize


class VQModelCTCAuxCondNoQ(pl.LightningModule):
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
        self.use_ctc_emb = ddconfig["use_ctc_emb"]
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        try:
            num_of_color_combinations=len(self.trainer.datamodule.datasets['train'].data.data.permutations)
        except:
            num_of_color_combinations = 729#len(self.trainer.datamodule.datasets['train'].data.data.permutations)
        self.decoder = DecoderCondAlsoOnTextPred(**ddconfig)
        self.decoder_text_map = DecoderCond(**ddconfig)

        self.loss = instantiate_from_config(lossconfig)
        self.color_pred_head = ColorPredictionHeadEnc(num_of_color_combinations)
        self.quant_conv  = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.encode_text = partial(t5_encode_text, name = DEFAULT_T5_NAME)
        self.attn_pool   = PerceiverResampler(input_dim=768,dim=512,depth=2,max_seq_len=64)
        self.ctc_head    = CTCAuxWrapper()
        self.ctc_head_predict_legend    = CTCAuxWrapper()
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            print(f"Loaded:{ckpt_path}")
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_counter = 0
        self.pretrain_ocr = pretrain_ocr


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        err=self.load_state_dict(sd, strict=False)
        print(f"Err loading:{err}")

    def forward(self, input_figure, prompt_emb=None,  titles=None, lop_concats=None):
        ###Encocde prompt
        if prompt_emb is not None:
            with torch.no_grad():
                text_embeds, text_masks = self.encode_text(prompt_emb, return_attn_mask=True)
            text_embeds, text_masks = map(lambda t: t.to(input_figure.device), (text_embeds, text_masks))
            text_embeds_perc = self.attn_pool(text_embeds,text_masks)
        ###Encocde Image
        encoder_output = self.encoder(input_figure)
        ##
        predicted_output_lop = self.ctc_head_predict_legend(encoder_output,text_embeds_perc)
        ###
        mid_layers = self.quant_conv(encoder_output)
        h = self.post_quant_conv(mid_layers)
        ###
        #Extract image with only text
        text_map,text_map_emb = self.decoder_text_map(h,text_emb=text_embeds_perc)#,title_embs = title_emb,legends_embs=legends_emb)
        # Extract edited image use text map_emb text_map is needed for supervision
        dec = self.decoder(h,text_map=text_map_emb,text_emb=text_embeds_perc)#,lop_concats=predicted_output_lop)#,title_embs = title_emb,legends_embs=legends_emb)

        ctc_loss = torch.tensor([0])
        return dec,text_map
        #color_comb_pred = self.color_pred_head(h)

        #ctc_loss,title_pair,legened_pair,logits_title, logits_legends,title_emb, legends_emb =self.ctc_head(quant,titles_emb,titles_emb_msak,lop_concats_emb,lop_concats_emb_mask)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()


    def training_step(self, batch, batch_idx, optimizer_idx):
        self.batch_counter+=1
        batch,meta_data,tensor_nonbin,out_seg_map = batch
        input_image = self.get_input(batch, "input_image")
        output_image = self.get_input(batch, "output_image")

        xrec, ctc_loss = self(input_image,batch["prompt"],tensor_nonbin)#,titles=meta_data["title"],lop_concats=meta_data["lop_concat"])
        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.loss( output_image, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(),  split="train")

            aeloss = aeloss#+0.1*ctc_loss
            log_dict_ae["ctc_loss"]=ctc_loss
            #self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
                # discriminator
            discloss, log_dict_disc = self.loss( output_image, xrec, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            #self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        #batch = batch
        batch,meta_data,tensor_nonbin,out_seg_map = batch
        input_image = self.get_input(batch, "input_image")
        output_image = self.get_input(batch, "output_image")
        xrec, ctc_loss = self(input_image,batch["prompt"])#,titles=meta_data["title"],lop_concats=meta_data["lop_concat"])
        aeloss, log_dict_ae  = self.loss( input_image, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val",target_images=output_image)

        discloss, log_dict_disc = self.loss( output_image, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        aeloss = aeloss #+ 0.1 * ctc_loss
        #log_dict_ae["ctc_loss"] = ctc_loss
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log('lr',self.trainer.lightning_optimizers[0].optimizer.param_groups[0]['lr'],
                 prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
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


    def configure_optimizers(self):
        lr = self.learning_rate
        if self.pretrain_ocr:
            opt_ae = torch.optim.Adam([{"params": list(self.encoder.parameters()) +
                                                  list(self.decoder.parameters()) +
                                                  list(self.quant_conv.parameters()) +
                                                  list(self.post_quant_conv.parameters()), "lr": 0},
                                       {"params": list(self.ctc_head.parameters())}],
                                      lr=lr, betas=(0.5, 0.9))
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=0, betas=(0.5, 0.9))
        else:
            opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                      list(self.decoder.parameters()) +
                                      list(self.quant_conv.parameters()) +
                                      list(self.post_quant_conv.parameters())+
                                      list(self.ctc_head.parameters()),
                                      lr=lr, betas=(0.5, 0.9))
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=0.1*lr, betas=(0.5, 0.9))
        milestones = list(len(self.trainer.datamodule.datasets['train'].data.data)*np.array([1,2,3]))

        scheduler_opt_ae   = torch.optim.lr_scheduler.MultiStepLR(opt_ae,milestones=milestones, gamma=0.3)
        scheduler_opt_disc = torch.optim.lr_scheduler.MultiStepLR(opt_ae,milestones=milestones, gamma=0.3)

        return [opt_ae, opt_disc], [scheduler_opt_ae,scheduler_opt_disc]
    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        if isinstance(batch,list):
            batch = batch[0]
        if self.image_key in batch:
            x = self.get_input(batch, self.image_key)
        else:
            output_image = self.get_input(batch, "output_image")
            input_image = self.get_input(batch, "input_image")
        input_image = input_image.to(self.device)
        xrec, _ = self(input_image,batch["prompt"])
        print("Logging Image")
        if output_image.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            output_image = self.to_rgb(output_image)
            xrec = self.to_rgb(xrec)
        log["inputs"] = torch.cat([input_image[0],output_image[0],xrec[0]],dim=-1)
        #log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x











