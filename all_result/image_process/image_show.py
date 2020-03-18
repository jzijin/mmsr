import cv2


path = "/home/jzijin/code/bysj/code/mmsr/datasets/test_x4_128/HR/X4/"
name = '3960.jpg'

img = cv2.imread(path+name)
cv2.imshow("test", img)
cv2.waitKey(0)
