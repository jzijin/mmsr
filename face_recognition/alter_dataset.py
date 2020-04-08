import os
from PIL import Image
import torch
from torch.utils import data
import lmdb
import numpy as np
from torchvision import transforms as T
import util as util
# from SR_model import IncNet
# import torchvision
# import cv2
# import sys


class Dataset(data.Dataset):

    def __init__(self, root="/home/jzijin/windows/f/celeba_bak/img_align_celeba.lmdb",  idenditys="/home/jzijin/code/bysj/code/mmsr/face_recognition/identity_CelebA.txt", phase='train'):
        self.phase = phase
        self.root = root
        self.paths, self.sizes = util.get_image_paths('lmdb', self.root)
        self.env = None
        assert self.paths, 'Error: path is empty'
        # print(self.paths, self.sizes)
        # exit()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(idenditys, 'r') as f:
            imgs = f.readlines()
        self.identity = {}
        for i in imgs:
            s = i.split()
            self.identity[s[0]] = np.int32(int(s[1])-1)
            # self.identity[s[0]] = s[1]
        # self.net = IncNet().to(self.device)
        # self.net.load_state_dict(torch.load("/home/jzijin/code/bysj/code/mmsr/experiments/008_IncNetx4_2/models/232000_G.pth", map_location=lambda storage, loc: storage))


        # self.hr_path = hr_path
        # self.sr_path = sr_path


        # self.imgs1 = os.listdir(hr_path)
        # self.imgs2 = os.listdir(sr_path)
        # exit()
        # imgs = [os.path.join(root, img[:-1]) for img in imgs]
        # self.imgs = np.random.permutation(imgs)

        # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                         std=[0.5, 0.5, 0.5])

        # normalize = T.Normalize(mean=[0.5], std=[0.5])

        # if self.phase == 'train':
            # self.transforms = T.Compose([
                # # T.RandomHorizontalFlip(),
                # T.ToTensor(),
                # # normalize
            # ])
        # else:
            # self.transforms = T.Compose([
                # # T.CenterCrop(self.input_shape[1:]),
                # T.ToTensor(),
                # # normalize
            # ])
    def _init_lmdb(self):
        self.env = lmdb.open(self.root, readonly=True, lock=False, readahead=False, meminit=False)



    def __getitem__(self, index):
        # img1_path = self.hr_path + self.imgs1[index]
        # img2_path = self.sr_path + self.imgs2[index]
        # img1 = Image.open(img1_path)
        # img2 = Image.open(img2_path)
        # img1 = self.transforms(img1)
        # img2 = self.transforms(img2)
        # label = np.int32(int(os.path.basename(img1_path)[:-4])-1)
        # # print(label)
        # return torch.cat([img1, img2], 0), label
        # # print(os.path.basename(img1_path))
        # print(os.path.basename(img2_path))

        # exit()
        self._init_lmdb()
        GT_path = self.paths[index]
        resolution = [int(s) for s in self.sizes[index].split('_')]
        img = util.read_img(self.env, GT_path, resolution)
        label = self.identity[GT_path]

        # sample = self.imgs[index]
        # splits = sample.split()
        # img_path = splits[0]
        # data = Image.open(img_path)
        # data = self.transforms(data)
        # data2 = self.net(data)
        # data = torch.cat([data, data2], 1)
        # print(data.size())
        # label = np.int32(int(splits[1])-1)

        mod_scale = 4
        up_scale = 4
        width = int(np.floor(img.shape[1] / mod_scale))
        height = int(np.floor(img.shape[0] / mod_scale))
        # modcrop
        if len(img.shape) == 3:
            image_HR = img[0:mod_scale * height, 0:mod_scale * width, :]
        else:
            image_HR = img[0:mod_scale * height, 0:mod_scale * width]

        # LR
        image_LR = util.imresize_np(image_HR, 1 / up_scale, True)
        # bic
        # image_Bic = util.imresize_np(image_LR, up_scale, True)
        img_GT = image_HR[:, :, [2, 1, 0]]
        img_LR = image_LR[:, :, [2, 1, 0]]

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()
        # print(img_GT.size())
        # print(img_LR.size())
        # print(img)
        # print(label)
        # exit(0)
        return img_LR, img_GT,  label
        # return data, label
        # print(index)
        # return 1, 1

    def __len__(self):
        return len(self.paths)

if __name__ == '__main__':
    # print(os.path.basename("/home/jzijin/aaa.jpg"))
    train_dataset = Dataset()
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=8,
                                  shuffle=True,
                                  num_workers=4)
    for j in trainloader:
        print(j)
