# from .models.archs.RRDBNet_arch import RRDBNet
# import modelsarch.RRDBNet_arch.RRDBNet as RRDBNet
import torch
from models.archs.RRDBNet_arch import RRDBNet
from collections import OrderedDict

# a = {'a':}




device = torch.device('cuda')
GAN_path = '/home/jzijin/windows/e/code/ubuntu_code/bysj/code/mmsr/experiments/pretrained_models/RRDB_ESRGAN_x4.pth'
psnr_path = '/home/jzijin/windows/e/code/ubuntu_code/bysj/code/mmsr/experiments/pretrained_models/RRDB_PSNR_x4.pth'
inter_path = '/home/jzijin/windows/e/code/ubuntu_code/bysj/code/mmsr/experiments/pretrained_models/RRDB_INTER90_x4.pth'
GAN_ = RRDBNet(in_nc=3, out_nc=3,nf=64, nb=23).to(device)
PSNR_ = RRDBNet(in_nc=3, out_nc=3,nf=64, nb=23).to(device)
inter_ = RRDBNet(in_nc=3, out_nc=3,nf=64, nb=23).to(device)


GAN_.load_state_dict(torch.load(GAN_path))
PSNR_.load_state_dict(torch.load(psnr_path))
inter_.load_state_dict(torch.load(psnr_path))


G = GAN_.state_dict()
P = PSNR_.state_dict()
inter_dict = inter_.state_dict()

alpha = 0.9
for k,v in G.items():
    tmp = alpha*v + (1-alpha) * P[k]
    inter_dict[k] = tmp

torch.save(inter_dict, inter_path)

