import cv2
import numpy as np
from image_registration import chi2_shift, cross_correlation_shifts


def generate_overlay(video_frames, width, height, fps, outputPath):
    print('Saving overlay result to', outputPath)
    frame_lists = sorted(video_frames, key=len, reverse=True)
    print('len', len(frame_lists))

    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outputPath, codec, fps / 2, (width, height))
    shifts = {}

    for idx, base_frame in enumerate(frame_lists[0]):
        # Overlay frames
        bg_frame = base_frame.frame
        for listIdx, frameList in enumerate(frame_lists[1:]):
            if(idx < len(frameList)):
                overlay_frame = frameList[idx]
            else:
                overlay_frame = frameList[len(frameList) - 1]

            alpha = 1.0 / (listIdx + 2)
            beta = 1.0 - alpha
            corrected_frame = image_registration(bg_frame, overlay_frame, shifts, listIdx, width, height)
            bg_frame = cv2.addWeighted(corrected_frame, alpha, bg_frame, beta, 0)

        # Draw the non-opacity balls
        for listIdx, frameList in enumerate(frame_lists[1:]):
            if(idx < len(frameList)):
                overlay_frame = frameList[idx]
            else:
                overlay_frame = frameList[len(frameList) - 1]
            if(overlay_frame.ball_in_frame):
                cv2.circle(bg_frame, overlay_frame.ball, 13, (255, 255, 255), -1)

        if(base_frame.ball_in_frame):
            cv2.circle(bg_frame, base_frame.ball, 13, (255, 255, 255), -1)

        result_frame = cv2.cvtColor(bg_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('result_frame', result_frame)
        out.write(result_frame)
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break


def image_registration(ref_image, offset_image, shifts, listIdx, width, height):
    if(listIdx not in shifts):
        xoff, yoff = cross_correlation_shifts(
            ref_image[:, :, 0], offset_image.frame[:, :, 0])
        shifts[listIdx] = (xoff, yoff)
    else:
        xoff, yoff = shifts[listIdx]

    offset_image.ball = tuple([offset_image.ball[0] - int(xoff), offset_image.ball[1] - int(yoff)])
    matrix = np.float32([[1, 0, -xoff], [0, 1, -yoff]])
    corrected_image = cv2.warpAffine(offset_image.frame, matrix, (width, height))

    return corrected_image
