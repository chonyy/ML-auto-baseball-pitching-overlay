import cv2
import numpy as np
import os

path = r'D:/gitrepo/ML-auto-baseball-pitching-overlay/img/image1.jpg'
img = cv2.imread(path)

height, width, _ = img.shape

pts = np.array([[200, 200], [250, 500], [400, 600], [
               550, 500], [600, 200]], dtype='int32')

# for pt in pts:
#     cv2.circle(img, tuple(pt), 20, (255, 0, 0), -1)

img = cv2.polylines(img, [pts], False, (255, 0, 0), 5, lineType=cv2.LINE_AA)

img = cv2.resize(img, (int(width * 0.5), int(height * 0.5)), fx=0.5, fy=0.5)
cv2.imshow('img', img)
cv2.waitKey(0)
