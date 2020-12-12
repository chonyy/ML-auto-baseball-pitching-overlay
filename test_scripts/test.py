import cv2
import pickle
import os

def generate_overlay(frames, width, height, fps):
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('test7.avi', codec, fps, (width, height))
    alpha = 0.5

    framesList = sorted(frames, key=len, reverse=True)

    for idx, frame in enumerate(framesList[0]):
        for frameList in framesList[1:]:
            if(idx < len(frameList)):
                frame = cv2.addWeighted(frameList[idx], alpha, frame, 1 - alpha, 0)
            else:
                frame = cv2.addWeighted(frameList[len(frameList) - 1], alpha, frame, 1 - alpha, 0)


        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', frame)
        out.write(frame)
        if cv2.waitKey(120) & 0xFF == ord('q'): break

video_path = 'videos/videos/11.mp4'
vid = cv2.VideoCapture(video_path)

width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS) / 4)
with open('frames7.pkl', 'rb') as f:
    frames = pickle.load(f)

generate_overlay(frames, width, height, fps)