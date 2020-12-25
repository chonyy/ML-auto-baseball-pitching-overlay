import time
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import copy
import time
from skimage.registration import phase_cross_correlation
from image_registration import cross_correlation_shifts, chi2_shift
from scipy.ndimage import shift
from src.SORT_tracker.tracker import Tracker
from src.SORT_tracker.sort import *


def generate_overlay(frames, width, height, fps, outputPath):
    print('Saving overlay result to', outputPath)
    frameLists = sorted(frames, key=len, reverse=True)

    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outputPath, codec, fps / 2, (width, height))
    shifts = {}

    for idx, baseFrame in enumerate(frameLists[0]):
        for listIdx, frameList in enumerate(frameLists[1:]):
            if(idx < len(frameList)):
                overlayFrame = frameList[idx]
            else:
                overlayFrame = frameList[len(frameList) - 1]

            alpha = 1.0 / (listIdx + 2)
            beta = 1.0 - alpha
            corrected_frame = image_registration(
                baseFrame, overlayFrame, shifts, listIdx, width, height)
            # baseFrame = cv2.addWeighted(baseFrame, 1, corrected_frame, alpha, 0)
            baseFrame = cv2.addWeighted(
                corrected_frame, alpha, baseFrame, beta, 0)

        resultFrame = cv2.cvtColor(baseFrame, cv2.COLOR_RGB2BGR)
        cv2.imshow('resultFrame', resultFrame)
        out.write(resultFrame)
        if cv2.waitKey(120) & 0xFF == ord('q'):
            break


def image_registration(ref_image, offset_image, shifts, listIdx, width, height):
    prev_time = time.time()

    if(listIdx not in shifts):
        xoff, yoff = cross_correlation_shifts(
            ref_image[:, :, 0], offset_image[:, :, 0])
        shifts[listIdx] = (xoff, yoff)
    else:
        xoff, yoff = shifts[listIdx]

    matrix = np.float32([[1, 0, -xoff], [0, 1, -yoff]])
    corrected_image = cv2.warpAffine(offset_image, matrix, (width, height))

    return corrected_image


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


def getBallFrames(video_path, input_size, infer, size, iou, score_threshold, tiny):
    print("Video from: ", video_path)
    vid = cv2.VideoCapture(video_path)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    frame_id = 0

    track_colors = [(161, 235, 52), (161, 235, 52), (161, 235, 52), (235, 171, 52), (255, 235, 52), (255, 235, 52), (255, 235, 52), (210, 235, 52), (52, 235, 131), (52, 64, 235), (0, 0, 255), (0, 255, 255),
                    (255, 0, 127), (127, 0, 127), (255, 127, 255), (127, 0, 255), (255, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 255), (50, 100, 150), (10, 50, 150), (120, 20, 220)]

    # Create Object Tracker
    tracker = Sort(max_age=8, min_hits=3, iou_threshold=0.3)
    balls = []
    ball_frames = []
    frames = []

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            frames.append(frame)
        else:
            if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Processing complete")
                break
            raise ValueError(
                "Something went wrong! Try with another video format")

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
            cv2.rectangle(frame, start, end, (255, 0, 0), 5)
            cv2.putText(frame, str(t[4]), start, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)

            clr = t[4] % 12
            centerX = int((t[0] + t[2]) / 2)
            centerY = int((t[1] + t[3]) / 2)
            balls.append([centerX, centerY, t[4]])

        # Draw the line
        if(len(balls) > 0):
            ball_points = copy.deepcopy(balls)
            for point in ball_points:
                clr = track_colors[point[2] % 12]
                del point[2]
            ball_points = np.array(ball_points, dtype='int32')
            cv2.polylines(frame, [ball_points], False, clr, 22, lineType=cv2.LINE_AA)

        # Store the frames with ball tracked
        if(len(trackings) > 0):
            if(len(ball_frames) == 0):
                ball_frames.extend(frames[-20:])
            ball_frames.append(frame)
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
