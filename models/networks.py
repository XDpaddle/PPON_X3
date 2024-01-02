import paddle
import models.archs.classSR_rcan_arch as classSR_rcan_arch
#import models.archs.dsrnet as dsrnet
import models.archs.PPON as PPON


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    scale = opt_net['scale']#!
    print('*'*100,opt_net)

    # image restoration
    if which_model == 'PPON':
        netG = PPON.PPON_content(upscale=scale)#!


    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG