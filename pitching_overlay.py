import tensorflow as tf
import os
import sys
import warnings
import logging
from tensorflow.python.saved_model import tag_constants
from optparse import OptionParser
from src.get_pitch_frames import get_pitch_frames
from src.generate_overlay import generate_overlay

# Ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
tf.get_logger().setLevel(logging.ERROR)

# Allow GPU memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == '__main__':
    optparser = OptionParser()
    optparser.add_option('-f', '--videos_folder',
                         dest='rootDir',
                         help='Root directory that contains your pitching videos',
                         default='./videos/videos1')
    (options, args) = optparser.parse_args()

    # Initialize variables
    size = 416
    iou = 0.45
    score = 0.5
    weights = './model/yolov4-tiny-baseball-416'

    # Load pretrained model
    saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    rootDir = options.rootDir
    outputPath = rootDir + '/Overlay.avi'
    # Store the pitch frames and ball location of each video
    pitch_frames = []

    # Iterate all videos in the folder
    for idx, path in enumerate(os.listdir(rootDir)):
        print(f'Processing Video {idx + 1}')
        video_path = rootDir + '/' + path
        try:
            ball_frames, width, height, fps = get_pitch_frames(video_path, infer, size, iou, score)
            pitch_frames.append(ball_frames)
        except Exception as e:
            print(f'Error: Sorry we could not get enough baseball detection from the video, video {path} will not be overlayed')
            print(e)

    if(len(pitch_frames)):
        generate_overlay(pitch_frames, width, height, fps, outputPath)
