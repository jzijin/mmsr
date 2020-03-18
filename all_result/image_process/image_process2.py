import cv2
import os
import numpy as np
SCALE = 4

GT_path = '/home/jzijin/code/bysj/code/mmsr/datasets/test_x4_128/HR/X4/'
VDSR_path = '/home/jzijin/code/bysj/code/mmsr/results/VDSR/test_x4_128/'

Inc_net = '/home/jzijin/windows/e/code/ubuntu_code/bysj/code/mmsr/results/IncNetx4/test_x4_128/'
# Inc_net_GAN = '/home/jzijin/windows/e/code/ubuntu_code/bysj/code/mmsr/results/IncNet_GAN/test_x4_128/'
Inc_net_GAN = '/home/jzijin/windows/e/code/ubuntu_code/bysj/code/mmsr/results/IncNet_GAN/test_x4_128_aaa/'
ESRGAN = '/home/jzijin/windows/e/code/ubuntu_code/bysj/code/mmsr/results/RRDB_ESRGAN_x4/test_x4_128/'
# Super_FAN_psnr = '/home/jzijin/code/bysj/code/mmsr/results/FAN_x4/bak/'
Super_FAN = '/home/jzijin/code/bysj/code/mmsr/results/FAN_x4/test_x4_128/'

# BIC_path = '/home/jzijin/windows/e/code/ubuntu_code/bysj/code/mmsr/datasets/test_celeba_x4/Bic/X4/'
BIC_path = '/home/jzijin/code/bysj/code/mmsr/datasets/test_x4_128/Bic/X4/'
save_path = './'
image_size_x = 128
image_size_y = 128

row = 5
col = 6

# image_names = ["3901","3904","3915", "3917", "3960"]
# all_images = []
# for i in image_names:
    # # fsr_net
    # all_images.append(GT_path+i+".jpg")
    # all_images.append(BIC_path+i+".jpg")
    # # all_images.append(Inc_net+i[:-3]+'png')
    # # all_images.append(ESRGAN+i[:-3]+'png')
    # # all_images.append(Super_FAN_psnr+i[:-3]+'png')
    # all_images.append(SRCNN_path+i+'.png')
    # all_images.append(ESPCN_path+i+'.png')
    # all_images.append(VDSR_path+i+'.png')
    # all_images.append(Inc_net+i+'.png')
# all_images.reverse()
# print(all_images)


def img_process(name, key_point):
    img = cv2.imread(name)
    w = img.shape[1]
    h = img.shape[0]
    large_img = np.zeros((h*2, w, 3), np.uint8)
    eye = img[key_point[1]:key_point[3], key_point[0]:key_point[2]]
    large_img[0:h, 0:w] = img
    # cv2.imshow("tmp", large_img)
    # cv2.waitKey(0)
    # exit()


    # cv2.imshow("tmp", eye)
    # cv2.waitKey(0)
    width = int(eye.shape[1] * SCALE)
    height = int(eye.shape[0] * SCALE)
    eye_resized = cv2.resize(eye, (width, height), interpolation=cv2.INTER_CUBIC)

    cv2.imshow("tmp", eye_resized)
    cv2.waitKey(0)

    large_img[h:2*h, 0:w] = eye_resized

    img[h-height:h, w-width:w] = eye_resized
    cv2.rectangle(large_img, (key_point[0], key_point[1]), (key_point[2], key_point[3]), (0, 0, 255), 1)
    # cv2.rectangle(img, (w-width, h-height), (w, h), (0,255,0), 2)
    cv2.imshow("tmp", large_img)
    cv2.waitKey(0)
    return large_img

if __name__ == '__main__':
    name = "3960"
    key_point = [50,72,50+32,72+32]
    # img = img_process(GT_path+"000050.jpg", key_point)
    # cv2.imwrite(GT_path+"tmp/" + "000050.jpg", img)
    for i in [GT_path, BIC_path, VDSR_path, Inc_net, ESRGAN, Super_FAN, Inc_net_GAN]:
        if not os.path.exists(i+"tmp/"):
            os.mkdir(i+"tmp/")
        if i is GT_path:
            open_name = i+name+'.jpg'
        elif i is BIC_path:
            open_name = i+name+'.jpg'
        else:
            open_name = i + name + '.png'
        img = img_process(open_name, key_point)
        cv2.imwrite(i+'tmp/'+name+".png", img)

