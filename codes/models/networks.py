import torch
import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.discriminator_vgg_arch as SRGAN_arch
import models.archs.RRDBNet_arch as RRDBNet_arch
import models.archs.ERRDBNet_arch as ERRDBNet_arch
import models.archs.EDVR_arch as EDVR_arch


import models.archs.cascadeNet_arch as cascadeNet_arch
import models.archs.SuperFAN_arch as SuperFAN_arch
import models.archs.SRIncNet_arch as SRIncNet_arch
import models.archs.ESPCN_arch as ESPCN_arch
import models.archs.VDSR_arch as VDSR_arch
import models.archs.SRCNN_arch as SRCNN_arch


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RRDBNet':
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    # video restoration
    elif which_model == 'EDVR':
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'])

    elif which_model == 'cascade':
        netG = cascadeNet_arch.Generator(scale_factor=opt_net['scale'])

    elif which_model == 'fan':
        netG = SuperFAN_arch.Generator()
    elif which_model == 'ERRDBNet':
        netG = ERRDBNet_arch.ERRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    elif which_model == 'IncNet':
        netG = SRIncNet_arch.IncNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    elif which_model == 'ESPCN':
        netG = ESPCN_arch.ESPCN(upscale_factor=opt_net['upscale'], in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'VDSR':
        netG = VDSR_arch.VDSR(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], base_channels=opt_net['base_channels'], num_residuals=opt_net['num_residuals'])
    elif which_model == 'SRCNN':
        netG = SRCNN_arch.SRCNN()
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
