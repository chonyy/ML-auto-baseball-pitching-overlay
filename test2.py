import cv2
from image_registration import cross_correlation_shifts, chi2_shift
from skimage import io
from scipy.ndimage import shift

img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')



height, width, _ = img1.shape

img2 = cv2.resize(img2, (width, height), interpolation = cv2.INTER_AREA)

alpha = 0.5

res = cv2.addWeighted(img2, alpha, img1, 1 - alpha, 0)

# cv2.imshow('res', res)
# cv2.waitKey(0) & 0xFF == ord('q')

# img1 = io.imread('image1.jpg')
# img2 = io.imread('image2.jpg')

img2 = cv2.resize(img2, (width, height), interpolation = cv2.INTER_AREA)

img1f = img1[:, :, 0]
img2f = img2[:, :, 0]

noise = 0.9

# xoff, yoff = cross_correlation_shifts(img1f, img2f)
xoff, yoff, exoff, eyoff = chi2_shift(img1f, img2f, noise, return_error=True, upsample_factor='auto')

print(-yoff, -xoff)

correct = shift(img2f, shift=(5, -25), mode='constant')
res = cv2.addWeighted(correct, alpha, img1f, 1 - alpha, 0)

cv2.imshow('correct', res)
cv2.waitKey(0) & 0xFF == ord('q')