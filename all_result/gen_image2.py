import PIL.Image as Image
import os
# GT_path = '/home/jzijin/windows/e/code/ubuntu_code/bysj/code/mmsr/datasets/test_celeba_x4/HR/X4/'
GT_path = '/home/jzijin/code/bysj/code/mmsr/datasets/test_x4_128/HR/X4/tmp/'
# SRCNN_path = '/home/jzijin/code/bysj/code/mmsr/results/SRCNN/celeba/tmp/'
# ESPCN_path = '/home/jzijin/code/bysj/code/mmsr/results/ESPCN/celeba/tmp/'
VDSR_path = '/home/jzijin/code/bysj/code/mmsr/results/VDSR/test_x4_128/tmp/'

Inc_net = '/home/jzijin/windows/e/code/ubuntu_code/bysj/code/mmsr/results/IncNetx4/test_x4_128/tmp/'
Inc_net_GAN = '/home/jzijin/windows/e/code/ubuntu_code/bysj/code/mmsr/results/IncNet_GAN/test_x4_128_aaa/tmp/'
ESRGAN = '/home/jzijin/windows/e/code/ubuntu_code/bysj/code/mmsr/results/RRDB_ESRGAN_x4/test_x4_128/tmp/'
# Super_FAN_psnr = '/home/jzijin/code/bysj/code/mmsr/results/FAN_x4/bak/'
Super_FAN = '/home/jzijin/code/bysj/code/mmsr/results/FAN_x4/test_x4_128/tmp/'

# BIC_path = '/home/jzijin/windows/e/code/ubuntu_code/bysj/code/mmsr/datasets/test_celeba_x4/Bic/X4/'
BIC_path = '/home/jzijin/code/bysj/code/mmsr/datasets/test_x4_128/Bic/X4/tmp/'
save_path = './'
image_size_x = 128
image_size_y = 128*2

row = 1
col = 7

image_names = [name for name in os.listdir(GT_path)]
print(image_names)
all_images = []
for i in image_names:
    # fsr_net
    all_images.append(GT_path+i)
    all_images.append(BIC_path+i)
    # all_images.append(Inc_net+i[:-3]+'png')
    # all_images.append(SRCNN_path+i)
    # all_images.append(ESPCN_path+i)
    all_images.append(VDSR_path+i)
    all_images.append(Inc_net+i)
    all_images.append(ESRGAN+i[:-3]+'png')
    all_images.append(Super_FAN+i[:-3]+'png')
    all_images.append(Inc_net_GAN+i)
all_images.reverse()
print(all_images)
# exit()
save_nums = len(image_names) // (row)
# print(image_names)

def image_conpose(name):
    to_image = Image.new('RGB', (col*image_size_x, row*image_size_y))
    for j in range(row):
        for i in range(col):
            print("procedure of the ", i, "th.")
            from_image = Image.open(all_images.pop())
            to_image.paste(from_image, (i*image_size_x, j*image_size_y))
    to_image.save(save_path+name+'.png')

if __name__ == '__main__':
    print(len(image_names))
    print(save_nums)
    for i in range(save_nums):
        image_conpose(str(i))
