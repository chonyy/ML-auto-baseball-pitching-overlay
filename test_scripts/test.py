import cv2
import pickle
import os
import time
from skimage.registration import phase_cross_correlation
from image_registration import cross_correlation_shifts, chi2_shift
from scipy.ndimage import shift

def image_registration(ref_image, offset_image):
    prev_time = time.time()
    xoff, yoff = cross_correlation_shifts(ref_image[:, :, 0], offset_image[:, :, 0])
    # shifts, error, diffphase = phase_cross_correlation(ref_image[:, :, 0], offset_image[:, :, 0])
    # xoff = -shifts[1]
    # yoff = -shifts[0]
    corrected_image = shift(offset_image, shift=(-yoff, -xoff, 0), mode='constant')
    print(-yoff, -xoff)
    print(time.time() - prev_time)
    return corrected_image

def generate_overlay(frames, width, height, fps):
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('test8.avi', codec, fps, (width, height))
    alpha = 0.5

    framesList = sorted(frames, key=len, reverse=True)

    for idx, baseFrame in enumerate(framesList[0]):
        for frameList in framesList[1:]:
            if(idx < len(frameList)):
                overlayFrame = frameList[idx]
            else:
                overlayFrame = frameList[len(frameList) - 1]

            corrected_image = image_registration(baseFrame, overlayFrame)            
            resultFrame = cv2.addWeighted(corrected_image, alpha, baseFrame, 1 - alpha, 0)

        resultFrame = cv2.cvtColor(resultFrame, cv2.COLOR_RGB2BGR)
        cv2.imshow('resultFrame', resultFrame)
        out.write(resultFrame)
        if cv2.waitKey(120) & 0xFF == ord('q'): break

video_path = 'videos/videos/11.mp4'
vid = cv2.VideoCapture(video_path)

width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS) / 4)
with open('frames6.pkl', 'rb') as f:
    frames = pickle.load(f)

generate_overlay(frames, width, height, fps)