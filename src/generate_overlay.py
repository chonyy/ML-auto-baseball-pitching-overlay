import cv2
import numpy as np
import copy
from image_registration import chi2_shift, cross_correlation_shifts


def generate_overlay(video_frames, width, height, fps, outputPath):
    print('Saving overlay result to', outputPath)
    frame_lists = sorted(video_frames, key=len, reverse=True)

    for frame_list in frame_lists:
        complement_lost_tracking(frame_list)

    balls_in_curves = [[] for i in range(len(frame_lists))]
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outputPath, codec, fps / 2, (width, height))
    shifts = {}

    for idx, base_frame in enumerate(frame_lists[0]):
        # Overlay frames
        background_frame = base_frame.frame.copy()
        for list_idx, frameList in enumerate(frame_lists[1:]):
            if(idx < len(frameList)):
                overlay_frame = frameList[idx]
            else:
                overlay_frame = frameList[len(frameList) - 1]

            alpha = 1.0 / (list_idx + 2)
            beta = 1.0 - alpha
            corrected_frame = image_registration(background_frame, overlay_frame, shifts, list_idx, width, height)
            background_frame = cv2.addWeighted(corrected_frame, alpha, background_frame, beta, 0)

            # Prepare balls to draw
            if(overlay_frame.ball_in_frame):
                balls_in_curves[list_idx+1].append([overlay_frame.ball[0], overlay_frame.ball[1], overlay_frame.ball_color])

        if(base_frame.ball_in_frame):
            balls_in_curves[0].append([base_frame.ball[0], base_frame.ball[1], base_frame.ball_color])

        # Emphasize base frame
        # background_frame = cv2.addWeighted(base_frame.frame, 0.5, background_frame, 0.5, 0)

        # Draw transparent curve and non-transparent balls
        for trajectory in balls_in_curves:
            background_frame = draw_ball_curve(background_frame, trajectory)

        result_frame = cv2.cvtColor(background_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('result_frame', result_frame)
        out.write(result_frame)
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break


def image_registration(ref_image, offset_image, shifts, list_idx, width, height):
    if(list_idx not in shifts):
        xoff, yoff = cross_correlation_shifts(
            ref_image[:, :, 0], offset_image.frame[:, :, 0])
        shifts[list_idx] = (xoff, yoff)
    else:
        xoff, yoff = shifts[list_idx]

    offset_image.ball = tuple([offset_image.ball[0] - int(xoff), offset_image.ball[1] - int(yoff)])
    matrix = np.float32([[1, 0, -xoff], [0, 1, -yoff]])
    corrected_image = cv2.warpAffine(offset_image.frame, matrix, (width, height))

    return corrected_image


def draw_ball_curve(frame, trajectory):
    trajectory_weight = 0.7
    temp_frame = frame.copy()

    if(len(trajectory)):
        ball_points = copy.deepcopy(trajectory)
        for point in ball_points:
            clr = point[2]
            del point[2]
        ball_points = np.array(ball_points, dtype='int32')
        cv2.polylines(temp_frame, [ball_points], False, clr, 22, lineType=cv2.LINE_AA)
        frame = cv2.addWeighted(temp_frame, trajectory_weight, frame, 1-trajectory_weight, 0)

        last_ball = tuple(trajectory[-1][:-1])
        cv2.circle(frame, tuple(last_ball), 13, (255, 255, 255), -1)
    return frame


def complement_lost_tracking(frame_list):
    balls_x = [frame.ball[0] for frame in frame_list if frame.ball_in_frame]
    balls_y = [frame.ball[1] for frame in frame_list if frame.ball_in_frame]

    curve = np.polyfit(balls_x, balls_y, 2)
    poly = np.poly1d(curve)

    lost_sections = []
    in_lost = False
    frame_count = 0

    for idx, frame in enumerate(frame_list):
        # print(idx, frame.ball)
        if(frame.ball_lost_tracking and frame_count == 0):
            in_lost = True
            lost_sections.append([])

        if(in_lost and not(frame.ball_lost_tracking)):
            in_lost = False
            frame_count = 0

        if(in_lost):
            lost_sections[-1].append(idx)
            frame_count += 1

    print('sections', lost_sections)

    for lost_idx in lost_sections:
        if(lost_idx):
            prev_frame = frame_list[lost_idx[0]-1]
            last_frame = frame_list[lost_idx[-1]+1]
            clr = prev_frame.ball_color

            lost = [frame_list[i] for i in lost_idx]

            diff = last_frame.ball[0] - prev_frame.ball[0]
            # print(last_frame.ball[0])
            # print(prev_frame.ball[0])
            speed = int(diff / (len(lost)+1))
            print('speed', speed)

            for idx, frame in enumerate(lost):
                x = prev_frame.ball[0] + (speed * (idx+1))
                y = int(poly(x))
                frame.ball_in_frame = True
                frame.ball = (x, y)
                frame.ball_color = clr
                print('Add', x, y)
