import colorsys
import copy
import random
import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.ndimage import shift

from src.FrameInfo import FrameInfo
from src.generate_overlay import generate_overlay, draw_ball_curve
from src.SORT_tracker.sort import *
from src.SORT_tracker.tracker import Tracker


# def get_bright_color():
#     h, s, l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
#     r, g, b = [int(256*i) for i in colorsys.hls_to_rgb(h, l, s)]
#     return [r, g, b]


def predict(infer, frame, input_size, iou, score_threshold):
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score_threshold
    )

    boxes = boxes.numpy()
    scores = scores.numpy()
    classes = classes.numpy()
    valid_detections = valid_detections.numpy()

    return boxes, scores, classes, valid_detections


def distance(x, y):
    temp = (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
    return temp ** (0.5)


def detected_to_tracked(detected, tracked, tracker_min_hits):
    distance_threshold = 100
    first_ball = tracked[0]
    for untracked in detected[-(tracker_min_hits+1):]:
        if(distance(untracked, first_ball) < distance_threshold):
            untracked.append(first_ball[2])
            tracked.append(untracked)


def add_new_tracked_to_frame(frames, tracked_balls, tracker_min_hits, clr):
    modify_frames = frames[-(tracker_min_hits+1):]
    balls_to_add = tracked_balls[-(tracker_min_hits+1):]
    balls_to_add_temp = copy.deepcopy(balls_to_add)

    for point in balls_to_add_temp:
        del point[2]
    balls_to_add_temp = np.array(balls_to_add_temp, dtype='int32')

    for idx, frame in enumerate(modify_frames):
        # print('Add to frame', [balls_to_add_temp[:idx+1]])
        # cv2.polylines(frame.frame, [balls_to_add_temp[:idx+1]], False, clr, 22, lineType=cv2.LINE_AA)
        # print('Add', tuple(balls_to_add[idx][:-1]))
        frames[-((tracker_min_hits+1)-idx)] = FrameInfo(frame.frame, True, tuple(balls_to_add[idx][:-1]), clr)


def getBallFrames(video_path, input_size, infer, size, iou, score_threshold, tiny):
    print("Video from: ", video_path)
    vid = cv2.VideoCapture(video_path)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    tracker_min_hits = 3
    frame_id = 0

    # track_colors = [(161, 235, 52), (161, 235, 52), (161, 235, 52), (235, 171, 52), (255, 235, 52), (255, 235, 52), (255, 235, 52), (210, 235, 52), (52, 235, 131), (52, 64, 235), (0, 0, 255), (0, 255, 255),
    #                 (255, 0, 127), (127, 0, 127), (255, 127, 255), (127, 0, 255), (255, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 255), (50, 100, 150), (10, 50, 150), (120, 20, 220)]
    track_colors = [(161, 235, 52), (83, 254, 92), (255, 112, 52), (255, 112, 52), (255, 235, 52), (255, 38, 38), (255, 235, 52), (210, 235, 52), (52, 235, 131), (52, 64, 235), (0, 0, 255), (0, 255, 255),
                    (255, 0, 127), (127, 0, 127), (255, 127, 255), (127, 0, 255), (255, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 255), (50, 100, 150), (10, 50, 150), (120, 20, 220)]

    # Create Object Tracker
    tracker = Sort(max_age=8, min_hits=tracker_min_hits, iou_threshold=0.3)
    detected_balls = []
    tracked_balls = []
    ball_frames = []
    frames = []

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            frames.append(FrameInfo(frame, False))
        else:
            if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Processing complete")
                break
            raise ValueError("Something went wrong! Try with another video format")

        boxes, scores, classes, valid_detections = predict(
            infer, frame, input_size, iou, score_threshold)

        frame_h, frame_w, _ = frame.shape
        detections = []
        offset = 30
        accuracyThreshold = 0.95

        for i in range(valid_detections[0]):
            score = scores[0][i]
            if(score > accuracyThreshold):
                coor = boxes[0][i]
                coor[0] = (coor[0] * frame_h)
                coor[2] = (coor[2] * frame_h)
                coor[1] = (coor[1] * frame_w)
                coor[3] = (coor[3] * frame_w)

                centerX = int((coor[1] + coor[3]) / 2)
                centerY = int((coor[0] + coor[2]) / 2)

                print(f'Baseball Detected ({centerX}, {centerY}), Confidence: {str(round(score, 2))}')
                # cv2.circle(frame, (centerX, centerY), 10, (255, 0, 0), -1)
                detected_balls.append([centerX, centerY])
                detections.append(np.array([coor[1]-offset, coor[0]-offset, coor[3]+offset, coor[2]+offset, score]))

        if(len(detections) > 0):
            trackings = tracker.update(np.array(detections))
        else:
            trackings = tracker.update()

        # Add the valid trackings to balls_list
        for t in trackings:
            t = t.astype('int32')
            t[0] = int(t[0])
            t[1] = int(t[1])
            t[2] = int(t[2])
            t[3] = int(t[3])
            start = (t[0], t[1])
            end = (t[2], t[3])
            # cv2.rectangle(frame, start, end, (255, 0, 0), 5)
            # cv2.putText(frame, str(t[4]), start, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)

            clr = track_colors[t[4] % 12]
            centerX = int((t[0] + t[2]) / 2)
            centerY = int((t[1] + t[3]) / 2)
            tracked_balls.append([centerX, centerY, track_colors[t[4] % 12]])

        # Draw the line
        # draw_ball_curve(frame, tracked_balls)

        # Store the frames with ball tracked
        if(len(trackings) > 0):

            # At first track from SORT
            if(len(ball_frames) == 0):
                last_tracked_frame = frame_id
                detected_to_tracked(detected_balls, tracked_balls, tracker_min_hits)
                print('clr', clr)
                add_new_tracked_to_frame(frames, tracked_balls, tracker_min_hits, clr)
                # Add prior 20 frames before the first ball
                ball_frames.extend(frames[-20:])

            # Add lost frames
            if(frame_id - last_tracked_frame > 1):
                print('Lost frames:', frame_id - last_tracked_frame)
                frames_to_add = frames[last_tracked_frame: frame_id]
                for ball_frame in frames_to_add:
                    ball_frame.ball_lost_tracking = True
                ball_frames.extend(frames_to_add)

            # Add balls
            last_ball = tuple(tracked_balls[-1][:-1])
            ball_frames.append(FrameInfo(frame, True, last_ball, clr))
            last_tracked_frame = frame_id

        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        detection = cv2.resize((result), (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("result", detection)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    # Add five more frames after catching
    ball_frames.extend(frames[last_tracked_frame: last_tracked_frame+5])
    return ball_frames, width, height, fps
