import sys
import torch
from collections import OrderedDict

alpha = float(sys.argv[1])
print(alpha)

net_PSNR_path = '/home/jzijin/code/bysj/code/mmsr/experiments/008_IncNetx4/models/415000_G.pth'
net_ESRGAN_path = '/home/jzijin/code/bysj/code/mmsr/experiments/009_IncNet_GAN/models/317000_G.pth'
net_interp_path = '/home/jzijin/code/bysj/code/mmsr/experiments/pretrained_models/SRIncNet_interp_{:02d}.pth'.format(int(alpha*10))

net_PSNR = torch.load(net_PSNR_path)
net_ESRGAN = torch.load(net_ESRGAN_path)
net_interp = OrderedDict()

print('Interpolating with alpha = ', alpha)

for k, v_PSNR in net_PSNR.items():
    v_ESRGAN = net_ESRGAN[k]
    net_interp[k] = (1 - alpha) * v_PSNR + alpha * v_ESRGAN

torch.save(net_interp, net_interp_path)
