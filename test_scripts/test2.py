import cv2
from image_registration import cross_correlation_shifts, chi2_shift
from skimage import io
from scipy.ndimage import shift
import numpy as np
from skimage.registration import phase_cross_correlation
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

img1 = cv2.imread('img/image1.jpg')
img2 = cv2.imread('img/image2.jpg')



height, width, _ = img1.shape

img2 = cv2.resize(img2, (width, height), interpolation = cv2.INTER_AREA)

alpha = 0.5

img2 = cv2.resize(img2, (width, height), interpolation = cv2.INTER_AREA)

img1f = img1[:, :, 0]
img2f = img2[:, :, 0]

orig = cv2.addWeighted(img2f, alpha, img1f, 1 -  alpha, 0)

noise = 0.9

# xoff, yoff, exoff, eyoff = chi2_shift(img1f, img2f, noise, return_error=True, upsample_factor='auto')
# print(-yoff, -xoff)
# shifted = cv2.addWeighted(correct, alpha, img1, 1 - alpha, 0)
xoff, yoff = cross_correlation_shifts(img1f, img2f)
print(-yoff, -xoff)
matrix = np.float32([[1, 0, -xoff], [0, 1, -yoff]])
correct = cv2.warpAffine(img2, matrix, (width, height))
shifted2 = cv2.addWeighted(correct, alpha, img1, 1 - alpha, 0)

from skimage.registration import phase_cross_correlation
shiftedd, error, diffphase = phase_cross_correlation(img1, img2)
xoff = -shiftedd[1]
yoff = -shiftedd[0]
print(-yoff, -xoff)
matrix = np.float32([[1, 0, -xoff], [0, 1, -yoff]])
correct = cv2.warpAffine(img2, matrix, (width, height))
shifted3 = cv2.addWeighted(correct, alpha, img1, 1 - alpha, 0)

res = np.concatenate((shifted3, shifted2), axis=1)
small = cv2.resize(res, (0,0), fx=0.5, fy=0.5) 

cv2.imshow('correct', small)
cv2.waitKey(0) & 0xFF == ord('q')