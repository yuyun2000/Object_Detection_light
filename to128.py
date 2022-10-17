import cv2
import os

list = os.listdir('./data/val/img')

for i in range(len(list)):
    img = cv2.imread('./data/val/img/%s'%(list[i]))
    img = cv2.resize(img,(128,128))
    cv2.imwrite('./data/val/img/%s'%(list[i]),img)
