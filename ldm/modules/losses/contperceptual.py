import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
from ldm.modules.diffusionmodules.model import Normalize


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

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h

import functools

class BlurPool3D(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0, padding_size=None):
        super(BlurPool3D, self).__init__()
        if isinstance(filt_size, (list,tuple)):
            assert len(filt_size) == 3
        else:
            assert isinstance(filt_size, int)
            filt_size = [filt_size] * 3
        self.filt_size = filt_size
        self.pad_off = pad_off
        if padding_size is None:
            self.pad_sizes = [int(1.*(filt_size[0]-1)/2), int(np.ceil(1.*(filt_size[0]-1)/2)), int(1.*(filt_size[1]-1)/2), int(np.ceil(1.*(filt_size[1]-1)/2)), int(1.*(filt_size[2]-1)/2), int(np.ceil(1.*(filt_size[2]-1)/2))]
            self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        else:
            if isinstance(padding_size, (list,tuple)):
                assert len(padding_size) == 3
            else:
                assert isinstance(padding_size, int)
                padding_size = [padding_size] * 3
            self.pad_sizes =[padding_size[0], padding_size[0], padding_size[1], padding_size[1], padding_size[2], padding_size[2]]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        filt = torch.Tensor(cal_filt_kernel(filt_size[0])[:, None, None] *  cal_filt_kernel(filt_size[1])[None, :, None] * cal_filt_kernel(filt_size[2])[None, None, :])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:,:].repeat((self.channels,1,1,1,1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride,::self.stride]
        else:
            return F.conv3d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def cal_filt_kernel(filt_size):
    if(filt_size==1):
        a = np.array([1.,])
    elif(filt_size==2):
        a = np.array([1., 1.])
    elif(filt_size==3):
        a = np.array([1., 2., 1.])
    elif(filt_size==4):    
        a = np.array([1., 3., 3., 1.])
    elif(filt_size==5):    
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size==6):    
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size==7):    
        a = np.array([1., 6., 15., 20., 15., 6., 1.])
    return a

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad3d
        
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad3d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad3d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer



class NLayerDiscriminator3D(nn.Module):
    """Defines a 3D PatchGAN discriminator based on Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False, use_groupnorm=False,  act_tanh=False, use_blur_pool = False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator3D, self).__init__()
        if use_groupnorm:
            norm_layer = partial(Normalize)
        elif not use_actnorm:
            norm_layer = nn.BatchNorm3d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d
        
        kw = 4
        padw = 1
        if use_blur_pool:
            sequence = [
                BlurPool3D(input_nc, filt_size=kw, stride=2, padding_size=padw),
                nn.Conv3d(input_nc, ndf, kernel_size=3, stride=1, padding=1), 
                nn.LeakyReLU(0.2, True)
                ]
        else:
            sequence = [
                nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
                nn.LeakyReLU(0.2, True)
                ]
        nf_mult = 1
        
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            kw = (3, 4, 4) if n == n_layers - 1 else 4
            if use_blur_pool:
                sequence += [
                    BlurPool3D(ndf * nf_mult_prev, filt_size=kw, stride=2, padding_size=padw),
                    nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]

        kw = (1, 4, 4)
        padw = (0, 1, 1)
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.Tanh() if act_tanh else nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False, use_groupnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if use_groupnorm:
            norm_layer = partial(Normalize)
        elif not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, enbale_logvar = False, use_groupnorm=False,
                 disc_num_layers=3, disc_in_channels=3, disc_ch = 64, disc_factor=1.0, disc_weight=1.0, loss_norm = 'l1',
                 rec_weight = 1.0, perceptual_weight=1.0, use_actnorm=False, disc_conditional=False, act_tanh=False,
                 disc_loss="hinge", disc_loss_scale = 1.0, disc_precision="fp16", nll_loss_mean=False, warm_d_weight_steps = 0,
                 enable_2d = False, enable_3d=True, use_blur_pool=False,
                 use_adaptive_gan=True, penalty_weight = 1., penalty_radius = 3.):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        assert enable_2d or enable_3d
        self.kl_weight = kl_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.disc_loss_scale = disc_loss_scale
        self.use_adaptive_gan = use_adaptive_gan
        self.warm_d_weight_steps = warm_d_weight_steps
        self.penalty_radius = penalty_radius
        self.penalty_weight = penalty_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init, requires_grad=enbale_logvar)
        
        if enable_3d:
            self.discriminator = NLayerDiscriminator3D(input_nc=disc_in_channels,
                                                 ndf=disc_ch,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 use_groupnorm=use_groupnorm,
                                                 act_tanh=act_tanh,
                                                 use_blur_pool=use_blur_pool,
                                                 ).apply(weights_init)
        if enable_2d:
            self.discriminator_2d = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ).apply(weights_init)
            
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.disc_precision = disc_precision
        self.nll_loss_mean = nll_loss_mean
        self.rec_weight = rec_weight
        self.loss_norm = loss_norm
        self.enable_2d = enable_2d
        self.enable_3d = enable_3d
        

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
        return d_weight, nll_grads, g_grads

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, global_step, last_layer=None, split="train"):

        loss_dict = {}
        b, c, t, h, w = inputs.shape
        if optimizer_idx == 0:
        
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs.transpose(2, 1).contiguous().reshape(-1, c, h, w), reconstructions.transpose(2, 1).contiguous().reshape(-1, c, h, w))
                p_loss = p_loss.reshape(b, t, 1, 1, 1).transpose(1, 2).contiguous() * self.perceptual_weight
                loss_dict.update({f'{split}/p_loss': p_loss.mean().clone().detach().item()})
                
            if self.loss_norm == 'l1':
                rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
                if self.perceptual_weight > 0:
                    rec_loss = rec_loss +  p_loss
                nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
                weighted_nll_loss = self.rec_weight * nll_loss
                nll_loss = torch.mean(nll_loss) 
                weighted_nll_loss = torch.mean(weighted_nll_loss) 
                
            elif self.loss_norm == 'l2':
                rec_loss = F.mse_loss(reconstructions.contiguous(), inputs.contiguous(), reduction='mean')
                if self.perceptual_weight > 0:
                    rec_loss = rec_loss + torch.mean(p_loss)
                nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
                weighted_nll_loss = self.rec_weight * nll_loss
    
            
            penalty = reconstructions - reconstructions.mean(dim=(-1, -2))[:, :, :, None, None] - self.penalty_radius * reconstructions.std(dim=(-1,-2))[:, :, :, None, None] 
            penalty_loss =  penalty.clamp(min=0).mean() * self.penalty_weight
            
            
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            
            g_loss = 0
            
            if self.enable_2d:
                logits_fake_2d = self.discriminator_2d(rearrange(reconstructions, 'b c t h w -> (b t) c h w').contiguous())
                g_loss -= torch.mean(logits_fake_2d) 

            if self.enable_3d:
                logits_fake = self.discriminator(reconstructions.contiguous())
                g_loss -= torch.mean(logits_fake)
                
            if split == "train":
                d_weight, nll_grads, g_grads = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                loss_dict.update({f'{split}/nll_grads': nll_grads.norm().clone().detach().item()})
                loss_dict.update({f'{split}/g_grads': g_grads.norm().clone().detach().item()})
                if self.use_adaptive_gan:
                    weighted_g_loss = d_weight * disc_factor * g_loss
                    loss_dict.update({f'{split}/d_weight': d_weight.mean().clone().detach().item()})
                else:
                    if self.warm_d_weight_steps > 0 and (global_step - self.discriminator_iter_start) < self.warm_d_weight_steps:
                        d_weight = float(global_step - self.discriminator_iter_start + 1) / (self.warm_d_weight_steps + 1) * self.discriminator_weight  
                    else:
                        d_weight =  self.discriminator_weight           
                    weighted_g_loss =  disc_factor * g_loss * d_weight
                    loss_dict.update({f'{split}/d_weight': d_weight})
                    
            else:
                weighted_g_loss =  disc_factor * g_loss

            kl_loss = posteriors.kl()
            kl_loss = torch.mean(kl_loss)
            weighted_kl_loss = self.kl_weight * kl_loss
            
            loss = weighted_nll_loss + weighted_kl_loss + weighted_g_loss + penalty_loss
                
            loss_dict.update({f'{split}/total_loss_ae': loss.mean().clone().detach().item()})
            loss_dict.update({f'{split}/nll_loss': nll_loss.mean().clone().detach().item()})
            loss_dict.update({f'{split}/weighted_nll_loss': weighted_nll_loss.mean().clone().detach().item()})
            loss_dict.update({f'{split}/penalty_loss': penalty_loss.mean().clone().detach().item()})
            loss_dict.update({f'{split}/rec_loss': rec_loss.mean().clone().detach().item()})
            loss_dict.update({f'{split}/kl_loss': kl_loss.mean().clone().detach().item()})
            loss_dict.update({f'{split}/weighted_kl_loss':  self.kl_weight * kl_loss.mean().clone().detach().item()})
            loss_dict.update({f'{split}/logvar': self.logvar.clone().detach().item()})
            loss_dict.update({f'{split}/g_loss': g_loss.mean().clone().detach().item()})
            loss_dict.update({f'{split}/weighted_g_loss': weighted_g_loss.mean().clone().detach().item()})
            loss_dict.update({f'{split}/disc_factor': disc_factor})
            
            return loss, loss_dict

        if optimizer_idx == 1:
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = 0
            if self.enable_2d:
                logits_real_2d = self.discriminator_2d(rearrange(inputs, 'b c t h w -> (b t) c h w').contiguous().detach())
                logits_fake_2d = self.discriminator_2d(rearrange(reconstructions, 'b c t h w -> (b t) c h w').contiguous().detach())
                d_loss += disc_factor * (self.disc_loss(logits_real_2d, logits_fake_2d).to(inputs.dtype))
                loss_dict.update({f'{split}/logits_fake_2d': logits_fake_2d.mean().clone().detach().item()})
                loss_dict.update({f'{split}/logits_real_2d': logits_real_2d.mean().clone().detach().item()})
            
            if self.enable_3d:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
                d_loss += disc_factor *  self.disc_loss(logits_real, logits_fake).to(inputs.dtype)
                loss_dict.update({f'{split}/logits_real': logits_real.mean().clone().detach().item()})
                loss_dict.update({f'{split}/logits_fake': logits_fake.mean().clone().detach().item()})
            loss_dict.update({f'{split}/disc_loss': d_loss.mean().clone().detach().item()})
        
            return d_loss, loss_dict

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

