import cv2
import copy
import numpy as np


def draw_ball_curve(frame, trajectory):
    trajectory_weight = 0.7
    temp_frame = frame.copy()

    if(len(trajectory)):
        ball_points = copy.deepcopy(trajectory)
        for point in ball_points:
            color = point[2]
            del point[2]
        ball_points = np.array(ball_points, dtype='int32')
        cv2.polylines(temp_frame, [ball_points], False, color, 22, lineType=cv2.LINE_AA)
        frame = cv2.addWeighted(temp_frame, trajectory_weight, frame, 1-trajectory_weight, 0)

        last_ball = tuple(trajectory[-1][:-1])
        cv2.circle(frame, tuple(last_ball), 13, (255, 255, 255), -1)
    return frame


def distance(x, y):
    temp = (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
    return temp ** (0.5)
