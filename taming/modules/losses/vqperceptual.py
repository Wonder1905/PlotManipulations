import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from taming.modules.losses.lpips import LPIPS, OCR_CRAFT_LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from main import save_img
import kornia

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight
def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss
def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss
class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

class VQLPIPSWithDiscriminatorOCRNoQ(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=0.2, mask_supervision_weight=1 ,ocr_perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.mask_supervision_weight = mask_supervision_weight

        # Definition of OCR perceptual losses
        self.ocr_perceptual_loss = OCR_CRAFT_LPIPS().eval() 
    
        self.ocr_perceptual_weight = ocr_perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self,  inputs, reconstructions,target_text_mask=None,pred_text_mask=None, optimizer_idx=0,
                global_step=0, last_layer=None, cond=None, metadata=None, split="train", target_images=None):
        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): disc_factor * d_loss.clone().detach().mean(),
                   #"{}/logits_real".format(split): logits_real.detach().mean(),
                   #"{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
        else:
            if split == 'test':
                self.perceptual_weight = 1 # Set this to one in the test set, to evaluate it

            importance_map = 100*torch.abs(inputs-target_images)+0.1 #such that there will be no zero elemnts
            importance_map_notwhite_in = torch.where(inputs != 1., 100., 0.)
            importance_map_notwhite_tar = torch.where(target_images != 1., 100., 0.)
            importance_map = importance_map + 0.1*importance_map_notwhite_in+0.1*importance_map_notwhite_tar
            importance_map = torch.clip(importance_map, 0, 100)
            importance_map_dila = kornia.morphology.dilation(importance_map, kernel=torch.ones(5, 5).cuda())
            rec_loss_map = importance_map_dila.detach()*torch.abs(target_images.contiguous() - reconstructions.contiguous())
            rec_loss =10* rec_loss_map.view(rec_loss_map.shape[0],-1).mean(-1)

            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                total_loss = rec_loss.squeeze() + self.perceptual_weight * p_loss.squeeze()
            else:
                p_loss = torch.tensor([0.0])

            if self.ocr_perceptual_weight > 0:
                p_ocr_loss = self.ocr_perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                total_loss = total_loss + self.ocr_perceptual_weight * p_ocr_loss.squeeze()
            else:
                p_ocr_loss = torch.tensor([0.0])
            if target_text_mask is not None:
                target_text_mask_64 = F.interpolate(target_text_mask.unsqueeze(1), size=(64, 64), mode='nearest').float()
                test_mask_loss = F.binary_cross_entropy(pred_text_mask,target_text_mask_64)#,weight=torch.tensor([1,10]).long())
            else:
                test_mask_loss = torch.tensor([0.0])
            total_loss += self.mask_supervision_weight*test_mask_loss


            total_loss = torch.mean(total_loss)

            # now the GAN part

            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(total_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = total_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/ocr_loss".format(split): self.ocr_perceptual_weight * p_ocr_loss.squeeze().mean(),
                   "{}/p_loss".format(split): self.perceptual_weight *  p_loss.detach().mean(),
                   "{}/mask_loss".format(split): self.mask_supervision_weight *  test_mask_loss.detach().mean(),
                   "{}/g_loss".format(split): disc_factor *d_weight *g_loss.detach().mean(),
                   }
            return loss, log



class VQLPIPSWithDiscriminatorOCR(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=0.2, ocr_perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        # Definition of OCR perceptual losses
        self.ocr_perceptual_loss = OCR_CRAFT_LPIPS().eval()

        self.ocr_perceptual_weight = ocr_perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, metadata=None, split="train", target_images=None):

        if split == 'test':
            self.perceptual_weight = 1  # Set this to one in the test set, to evaluate it
        if self.less4blank_loss:
            if target_images is not None:
                importance_map = 100 * torch.abs(
                    inputs - target_images) + 0.1  # such that there will be no zero elemnts
                importance_map_notwhite_in = torch.where(inputs != 1., 100., 0.)
                importance_map_notwhite_tar = torch.where(target_images != 1., 100., 0.)
                importance_map = importance_map + 0.1 * importance_map_notwhite_in + 0.1 * importance_map_notwhite_tar
                importance_map = torch.clip(importance_map, 0, 100)
                # plt.imshow(importance_map.squeeze(0).repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy())
                importance_map_dila = kornia.morphology.dilation(importance_map, kernel=torch.ones(5, 5).cuda())
            else:
                target_images = inputs
                importance_map = torch.where(inputs != 1., 1., 0.) + 0.001
                importance_map = torch.clip(importance_map.sum(1).unsqueeze(1), 0, 1)
                # plt.imshow(importance_map.squeeze(0).repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy())
                importance_map_dila = kornia.morphology.dilation(importance_map, kernel=torch.ones(3, 3).cuda())
                if metadata is not None and False:
                    coord = metadata["cordinates"]
                    # plt.imshow(inputs[0].permute(1, 2, 0).cpu().numpy())
                    # plt.imshow(inputs[0, :, int(384 * coord["y0"] / 480):int(384 * coord["y1"] / 480),
                    #           int(384 * coord["x0"] / 640):int(384 * coord["x1"] / 640)].permute(1, 2, 0).cpu().numpy())
                    # plt.imshow(inputs[0, :, coord["y1"]:coord["y0"], coord["x0"] : coord["x1"]].permute(1, 2, 0).cpu().numpy())
                    importance_map_focus_legend = torch.where(inputs != 1., 0., 0.)
                    importance_map_focus_legend[0, :, coord["y1"]:coord["y0"], coord["x0"]: coord["x1"]] = 0.5
                    importance_map_dila = 3 * importance_map_focus_legend + importance_map_dila
            rec_loss_map = importance_map_dila.detach() * torch.abs(
                target_images.contiguous() - reconstructions.contiguous())
            rec_loss = 10 * rec_loss_map.view(rec_loss_map.shape[0], -1).mean(-1)
        else:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            total_loss = rec_loss.squeeze() + self.perceptual_weight * p_loss.squeeze()
        else:
            p_loss = torch.tensor([0.0])

        if self.ocr_perceptual_weight > 0:
            p_ocr_loss = self.ocr_perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            total_loss = total_loss + self.ocr_perceptual_weight * p_ocr_loss.squeeze()
        else:
            p_ocr_loss = torch.tensor([0.0])

        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        total_loss = torch.mean(total_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(total_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = total_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): self.codebook_weight * codebook_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/ocr_loss".format(split): self.ocr_perceptual_weight * p_ocr_loss.squeeze().mean(),
                   "{}/p_loss".format(split): self.perceptual_weight * p_loss.detach().mean(),
                   "{}/g_loss".format(split): disc_factor * d_weight * g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): disc_factor * d_loss.clone().detach().mean(),
                # "{}/logits_real".format(split): logits_real.detach().mean(),
                # "{}/logits_fake".format(split): logits_fake.detach().mean()
            }
            return d_loss, log