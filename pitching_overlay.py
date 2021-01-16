from tensorflow.python.saved_model import tag_constants
from optparse import OptionParser
from src.utils import get_ball_frames, generate_overlay
import os
import sys
import warnings
import tensorflow as tf

# Ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Allow GPU memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == '__main__':
    optparser = OptionParser()
    optparser.add_option('-f', '--videos_folder',
                         dest='rootDir',
                         help='Root directory that contains your pitching videos',
                         default='./videos/videos')
    (options, args) = optparser.parse_args()

    # Initialize variables
    tiny = True
    size = 416
    iou = 0.45
    score = 0.5

    if(tiny):
        weights = './model/yolov4-tiny-baseball-416'
    else:
        weights = './model/yolov4-custom-416'

    # Load pretrained model
    saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    video_frames = []
    rootDir = options.rootDir
    outputPath = rootDir + '/Overlay.avi'

    # Iterate all videos in the folder
    for idx, path in enumerate(os.listdir(rootDir)):
        print(f'Processing Video {idx + 1}')
        ball_frames, width, height, fps = get_ball_frames(rootDir + '/' + path, infer, size, iou, score)
        video_frames.append(ball_frames)

    generate_overlay(video_frames, width, height, fps, outputPath)
