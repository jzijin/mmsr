from __future__ import print_function
from torch.utils import data
from dataset import Dataset
import os
import torch
from model import *
import torchvision
import numpy as np
import time
from config import Config
from torch.optim.lr_scheduler import StepLR
from test import *
from torchvision.models.resnet import resnet34

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


    criterion = FocalLoss(gamma=2)
    model = resnet34(pretrained=True)
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.fc = nn.Linear(512, 512, bias=True)

    metric_fc = ArcMarginProduct(1000, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    # model.load_state_dict(torch.load(opt.pretrain_model))
    print(model)
    print(metric_fc)
    model.to(device)
    metric_fc.to(device)
    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                                                    lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)
    start = time.time()
    for i in range(opt.start_epoch, opt.max_epoch+1):
        scheduler.step()
        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            # change the label to long type
            label = label.to(device).long()
            # get the the image feature
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
                start = time.time()
            if iters % opt.save_interval == 0 or i == opt.max_epoch:
                save_model(model, opt.checkpoints_path, opt.backbone, i, iters)
                model.eval()
                acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)


