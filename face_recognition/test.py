from __future__ import print_function
import os
import cv2
from model import *
import torch
import numpy as np
import time
from config import Config
from torch.nn import DataParallel
from torchvision.models.resnet import resnet18, resnet34
# from torchvision.trainsform import ToTensor()


def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def load_image(img_path):
    image = cv2.imread(img_path)
    # print(image.shape)
    # exit()
    if image is None:
        return None
    # print(np.fliplr(image))
    # ?????
    # image = np.dstack((image, np.fliplr(image)))
    # print(image.shape)
    # exit()
    # ?(128,128,2) -> (2,128,128)
    image = image.transpose((2, 0, 1))
    # (2,128,128) -> (2, 1, 128, 128)
    # image = image[:, np.newaxis, :, :]
    image = image[np.newaxis, : , :, :]
    # (2,3,128,128)
    # image = np.concatenate((image, image), axis=0)
    image = image.astype(np.float32, copy=False)
    # print(image.shape)
    # image -= 127.5
    image /= 255
    # print(image.shape)
    return image


"""
get a batchSize feature
"""
def get_featurs(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        # print(image.shape)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)
        # print(images.shape)
        # exit()

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            # 将一张图片的输出512的feature复制一份堆叠成1024
            # 我不知道为什么要这么做。
            cnt += 1

            data = torch.from_numpy(images)
            # print(data.size())
            # data = torch.cat([data, data], 1)
            # print(data.size())
            # exit(0)
            data = data.to(torch.device("cuda"))
            # print(data.size())
            output = model(data)
            # print(output.size())
            # feed into the model and get feature
            feature = output.data.cpu().numpy()
            # print(output.shape)

            # fe_1 = output[::2]
            # print(fe_1.shape)
            # fe_2 = output[1::2]
            # print(fe_2.shape)
            # exit()

            # (4,512)->(2,1024) ???????????
            # feature = np.hstack((fe_1, fe_2))

            if features is None:
                features = feature
            else:
                # vertical stack
                features = np.vstack((features, feature))
            # print(features.shape)
            # print(feature.shape)
            # exit()
            # exit()

            images = None

    # print(features.shape)
    # exit()
    # print(features.shape)

    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


# ?????????????
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    # print(y_score, y_true)
    # exit()
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        # ??????????
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        # print(acc)
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        # first iamge feature
        fe_1 = fe_dict[splits[0]]
        # print(fe_1.shape)
        # second image second feature
        fe_2 = fe_dict[splits[1]]
        # the label correct1 error0
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)
        # print(sim)
        # exit()

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, batch_size=batch_size)
    # print(len(identity_list))
    # print(features.shape)
    # exit()
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    # print(fe_dict)
    # exit()
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc


if __name__ == '__main__':

    opt = Config()
    # if opt.backbone == 'resnet18':
        # model = resnet_face18(opt.use_se)
    # elif opt.backbone == 'resnet34':
        # model = resnet34()
    # elif opt.backbone == 'resnet50':
        # model = resnet50()

    model = resnet34(pretrained=True)
    # model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.fc = nn.Linear(512, 512, bias=True)
    # print(model)
    # model = DataParallel(model)
    # load_model(model, opt.test_model_path)
    # model.load_state_dict(torch.load(opt.test_model_path))
    model.load_state_dict(torch.load("checkpoints/resnet18_1.pth"))
    model.to(torch.device("cuda"))

    # get all identity to a list
    identity_list = get_lfw_list(opt.lfw_test_list)
    # get all identity's path
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]
    # print(img_paths)
    # exit()

    model.eval()
    lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
    # lfw_test(model, img_paths, identity_list, 'tmp.txt', opt.test_batch_size)
    # lfw_test(model, img_paths, identity_list, 'tmp.txt', 1)





