import cv2
from main import my_detector
from norfair import Detection, Tracker, Video, draw_tracked_objects
from ultralytics import YOLO
from typing import List
import pdb
import numpy as np
from draw import center, draw
from yolo import YOLO, yolo_detections_to_norfair_detections

from norfair import AbsolutePaths, Paths, Tracker, Video
from norfair.camera_motion import HomographyTransformationGetter, MotionEstimator
from norfair.distances import create_normalized_mean_euclidean_distance

DISTANCE_THRESHOLD_CENTROID: float = 0.08




# detector = MyDetector()  # Set up a detector
video = Video(input_path="4k_Video.mp4")
tracker = Tracker(distance_function="euclidean", distance_threshold=100)

transformations_getter = HomographyTransformationGetter()

for frame in video:
   yolo_detections = my_detector(frame)

   norfair_detections = [Detection(points) for points in yolo_detections]
   tracked_objects = tracker.update(detections=norfair_detections)
   draw_tracked_objects(frame, tracked_objects)
   video.write(frame)
