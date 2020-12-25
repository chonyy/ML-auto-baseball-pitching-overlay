import cv2
import numpy as np

path = './image1.jpg'

img = cv2.imread(path)

print(img)

cv2.imshow('img', img)
cv2.waitKey(0)
