from __future__ import print_function
from torch.utils import data
from alter_dataset import Dataset
import os
import torch
from model import *
from SR_model import IncNet
# import torchvision
import numpy as np
import time
from math import log10
from config import Config
from torch.optim.lr_scheduler import StepLR
from test import *
from torchvision.models.resnet import resnet18

def save_model(model, save_path, name, iter_cnt, ii):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '_iters_' + str(ii)+ '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


if __name__ == '__main__':
    opt = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = Dataset()
    trainloader = data.DataLoader(train_dataset,
                                  batch_size = opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]


    criterionF = FocalLoss(gamma=2)
    criterion = nn.L1Loss()

    sr_model = IncNet().to(device)

    fr_model = resnet34().to(device)

    sr_model.load_state_dict(torch.load(opt.pretrain_model_G))
    fr_model.load_state_dict(torch.load(opt.pretrain_model_F))
    # fr_model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # fr_model.fc = nn.Linear(512, 512, bias=True)

    metric_fc = ArcMarginProduct(1000, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin).to(device)
    print(fr_model)
    print(metric_fc)
    # fr_model.to(device)
    # metric_fc.to(device)
    optimizerF = torch.optim.SGD([{'params': fr_model.parameters()}, {'params': metric_fc.parameters()}],
                                                                    lr=opt.lr, weight_decay=opt.weight_decay)
    sr_optim_params = []
    for k, v in sr_model.named_parameters():
        if v.requires_grad:
            sr_optim_params.append(v)
    optimizerG = torch.optim.Adam(sr_optim_params, lr=opt.lr_G, weight_decay=opt.wd_G, betas=(opt.beta1_G, opt.beta2_G))
    scheduler1 = StepLR(optimizerF, step_size=opt.lr_step, gamma=0.1)
    scheduler2 = StepLR(optimizerG, step_size=opt.lr_step_G, gamma=0.1)
    start = time.time()
    for i in range(opt.start_epoch, opt.max_epoch+1):
        scheduler1.step()
        scheduler2.step()
        fr_model.train()
        for ii, data in enumerate(trainloader):
            data_lr, data_input, label = data
            data_input = data_input.to(device)
            data_lr = data_lr.to(device)
            # change the label to long type
            label = label.to(device).long()
            data_sr = sr_model(data_lr)

            ### update face recognition model ###
            for p in fr_model.parameters():
                p.require_grad = False

            optimizerF.zero_grad()
            # get the the image feature
            feature = fr_model(data_sr.detach())
            output = metric_fc(feature, label)
            loss = criterionF(output, label)
            loss.backward()
            optimizerF.step()

            for p in fr_model.parameters():
                p.require_grad = False

            ### update face hallucination model ###
            optimizerG.zero_grad()
            loss_G = criterion(data_sr, data_input)
            real_f = fr_model(data_input)
            fake_f = fr_model(data_sr)
            loss_F = criterion(real_f, fake_f)
            sr_loss = loss_G + opt.alpha * loss_F
            sr_loss.backward()
            optimizerG.step()

            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                # print(output)
                output = np.argmax(output, axis = 1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time()-start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
                print('sr_loss {} fr_loss {}'.format(sr_loss.item(), loss.item()))
                start = time.time()
            if iters % opt.save_interval == 0:
                save_model(fr_model, opt.checkpoints_path, opt.backbone, i, iters)
                save_model(fr_model, opt.checkpoints_path, 'SRIncNet', i,iters)
                fr_model.eval()
                avg_psnr = 0.0
                index = 0
                with torch.no_grad():
                    val_images = []
                    for lr, hr, l in trainloader:
                        index += 1
                        if(index > 10):
                            break
                        sr = sr_model(lr.to(device))
                        mse = criterion(sr, hr.to(device))
                        psnr = 10 * log10(1 / mse.item())
                        avg_psnr += psnr
                print('===>avg psnr: {:.4f} dB'.format(avg_psnr / (opt.train_batch_size*index)))
                acc = lfw_test(fr_model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
