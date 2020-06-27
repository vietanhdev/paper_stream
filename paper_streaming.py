import numpy as np
import cv2
import os
import time
from distutils import util
import argparse
import time
import urllib
import _thread
from libs.paper_processor.paper_processor import PaperProcessor
from libs.hand_remover.hand_remover import HandRemover
from libs.stroke_filter.stroke_filter import StrokeFilter
from libs.webcam import pyfakewebcam
from threading import Lock
from libs.utils.common import *
from libs.config import *

OUTPUT_SIMULATED_CAMERA = False

paper_processor = PaperProcessor(REFERENCE_ARUCO_IMAGE_PATH, aruco_remove_mask_path=REFERENCE_ARUCO_REMOVE_IMAGE_PATH, smooth=True, debug=False, output_video_path=None)
hand_remover = HandRemover()
stroke_filter = StrokeFilter()

# create output video stream
if OUTPUT_SIMULATED_CAMERA:
    output_width, output_height = paper_processor.get_output_size()
    camera = pyfakewebcam.FakeWebcam(get_camera_path("PaperStreamCam"), output_width, output_height)
    
cap = cv2.VideoCapture("http://192.168.43.1:8080/video")
frame = None
frame_mutex = Lock()
    
# Define a function for the thread
def read_camera_thread():
    global cap, frame, frame_mutex
    while(True):
        frame_mutex.acquire()
        _, frame = cap.read()
        frame_mutex.release()
        time.sleep(0.05)

_thread.start_new_thread( read_camera_thread, () )
while(True):
    
    image = None
    frame_mutex.acquire()
    try:
        image = frame.copy()
    except:
        pass
    frame_mutex.release()
    
    if image is None:
        continue
        
    # get paper image
    start_time = time.time()
    is_cropped, processed_image = paper_processor.get_paper_image(image)
    print("Paper transform. time: {}".format(time.time() - start_time))
    
    # remove hand
    start_time = time.time()
    processed_image = hand_remover.process(processed_image, is_cropped=is_cropped)
    print("Hand removal time: {}".format(time.time() - start_time))
    
    # post processing
    start_time = time.time()
    processed_image = stroke_filter.process(processed_image)
    print("Post processing time: {}".format(time.time() - start_time))
        
    if OUTPUT_SIMULATED_CAMERA:
        if is_cropped:
            camera_frame = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            camera.schedule_frame(camera_frame)
    
    cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Debug",  frame)
    cv2.imshow("Result",  processed_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()