import cv2
import pickle
import os

# video_path = 'data/good.mp4'
# video_path2 = 'data/good2.mp4'

# vid = cv2.VideoCapture(video_path)
# vid2 = cv2.VideoCapture(video_path2)

# while True:
#         return_value, frame = vid.read()
#         return_value2, frame2 = vid2.read()
#         if not (return_value or return_value2):
#             break
#         alpha = 0.5  # Transparency factor.
#         frame = cv2.addWeighted(frame2, alpha, frame, 1 - alpha, 0)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'): break


video_path = 'data/good.mp4'
vid = cv2.VideoCapture(video_path)

width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS) / 4)
codec = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test7.avi', codec, fps, (width, height))

alpha = 0.5
frames = None
with open('frames7.pkl', 'rb') as f:
    frames = pickle.load(f)

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