import numpy as np
import cv2
import os
import time
from distutils import util
import argparse
import time
import urllib 
from libs.paper_processor.paper_processor import PaperProcessor
from libs.hand_remover.hand_remover import HandRemover
from libs.stroke_filter.stroke_filter import StrokeFilter
from libs.webcam import pyfakewebcam
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
while(True):
    
    ret, frame = cap.read()
    if ret == False:
        break
    if frame is not None:
        
        # get paper image
        is_cropped, processed_image = paper_processor.get_paper_image(frame)
        
        # remove hand
        processed_image = hand_remover.process(processed_image, is_cropped=is_cropped)
        
        # post processing 
        processed_image = stroke_filter.process(processed_image)
            
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