import cv2
import numpy as np
img1 = cv2.imread("0.png")
img2 = cv2.imread("1.png")
img3 = cv2.imread("3.png")
img4 = cv2.imread("2.png")
img5 = cv2.imread("4.png")
width = img1.shape[1]
height = img1.shape[0]
img6 = np.zeros((height*5, width, 3), np.uint8)
img6[0:height, 0:width] = img1
img6[height:height*2, 0:width] = img2
img6[height*2:height*3, 0:width] = img3
img6[height*3:height*4, 0:width] = img4
img6[height*4:height*5, 0:width] = img5
cv2.imwrite("aaaaa.png", img6)
