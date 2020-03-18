import PIL.Image as Image
import os


GT_path = '/home/jzijin/code/bysj/code/mmsr/datasets/test_x4_128/HR/X4/tmp/'
alpha_0 = '/home/jzijin/code/bysj/code/mmsr/results/IncNet_GAN/test_x4_128_alpha_0/tmp/'
alpha_2 = '/home/jzijin/code/bysj/code/mmsr/results/IncNet_GAN/test_x4_128_alpha_02/tmp/'
alpha_4 = '/home/jzijin/code/bysj/code/mmsr/results/IncNet_GAN/test_x4_128_alpha_04/tmp/'
alpha_6 = '/home/jzijin/code/bysj/code/mmsr/results/IncNet_GAN/test_x4_128_alpha_06/tmp/'
alpha_8 = '/home/jzijin/code/bysj/code/mmsr/results/IncNet_GAN/test_x4_128_alpha_08/tmp/'
alpha_10 = '/home/jzijin/code/bysj/code/mmsr/results/IncNet_GAN/test_x4_128_alpha_10/tmp/'
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
    all_images.append(alpha_0+i)
    all_images.append(alpha_2+i)
    all_images.append(alpha_4+i)
    all_images.append(alpha_6+i)
    all_images.append(alpha_8+i)
    all_images.append(alpha_10+i)
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
