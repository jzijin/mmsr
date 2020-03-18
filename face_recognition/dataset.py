import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys


class Dataset(data.Dataset):

    def __init__(self, hr_path="/home/jzijin/code/bysj/code/mmsr/datasets/DIV2K/DIV2K800_sub/",sr_path="/home/jzijin/code/bysj/code/mmsr/datasets/DIV2K/DIV2K800_sub_SR/",  phase='train'):
        self.phase = phase
        self.hr_path = hr_path
        self.sr_path = sr_path


        self.imgs1 = os.listdir(hr_path)
        self.imgs2 = os.listdir(sr_path)
        # exit()
        # imgs = [os.path.join(root, img[:-1]) for img in imgs]
        # self.imgs = np.random.permutation(imgs)

        # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                         std=[0.5, 0.5, 0.5])

        # normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                # T.RandomHorizontalFlip(),
                T.ToTensor(),
                # normalize
            ])
        else:
            self.transforms = T.Compose([
                # T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                # normalize
            ])

    def __getitem__(self, index):
        img1_path = self.hr_path + self.imgs1[index]
        img2_path = self.sr_path + self.imgs2[index]
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        img1 = self.transforms(img1)
        img2 = self.transforms(img2)
        label = np.int32(int(os.path.basename(img1_path)[:-4])-1)
        # print(label)
        return torch.cat([img1, img2], 0), label
        # print(os.path.basename(img1_path))
        # print(os.path.basename(img2_path))

        # exit()

        # sample = self.imgs[index]
        # splits = sample.split()
        # img_path = splits[0]
        # data = Image.open(img_path)
        # data = data.convert('L')
        # data = self.transforms(data)
        # label = np.int32(splits[1])
        # return data.float(), label
        # print(index)
        # return 1, 1

    def __len__(self):
        return len(self.imgs1)

if __name__ == '__main__':
    print(os.path.basename("/home/jzijin/aaa.jpg"))
    train_dataset = Dataset()
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=8,
                                  shuffle=True,
                                  num_workers=4)
    for i in trainloader:
        print(i)
