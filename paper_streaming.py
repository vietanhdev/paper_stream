import argparse
import os
import sys
import time
import urllib
from distutils import util
from threading import Lock

import cv2
import numpy as np
from PyQt5 import QtWidgets, uic

import _thread
from libs.config import *
from libs.hand_remover.hand_remover import HandRemover
from libs.paper_processor.paper_processor import PaperProcessor
from libs.stroke_filter.stroke_filter import StrokeFilter
from libs.utils.common import *
from libs.utils.ui_utils import *
from libs.webcam import pyfakewebcam

OUTPUT_SIMULATED_CAMERA = False

paper_processor = PaperProcessor(REFERENCE_ARUCO_IMAGE_PATH, aruco_remove_mask_path=REFERENCE_ARUCO_REMOVE_IMAGE_PATH, smooth=True, debug=False, output_video_path=None)
hand_remover = HandRemover()
stroke_filter = StrokeFilter()

# create output video stream
if OUTPUT_SIMULATED_CAMERA:
    output_width, output_height = paper_processor.get_output_size()
    camera = pyfakewebcam.FakeWebcam(get_camera_path("PaperStreamCam"), output_width, output_height)
    
frame = None
frame_mutex = Lock()
    
# camera reading thread
new_camera_url = None
def camera_reading_thread():
    global frame, frame_mutex, new_camera_url
    cap = cv2.VideoCapture(DEFAULT_WEBCAM_URL)
    while(True):
        
        if new_camera_url is not None:
            print("Opening new camera at: {}".format(new_camera_url))
            try:
                cap.release()
            except:
                pass
            cap = cv2.VideoCapture(new_camera_url)
            new_camera_url = None
        
        frame_mutex.acquire()
        ret, frame = cap.read()
        frame_mutex.release()

        if ret:
            window.cameraStatusLabel.setText("Camera status: Connected")
        else:
            window.cameraStatusLabel.setText("Camera status: Disconnected")
        time.sleep(0.05)

# processing thread    
def processing_thread():
    global frame, frame_mutex
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
        # print("Paper transform. time: {}".format(time.time() - start_time))
        
        # remove hand
        start_time = time.time()
        processed_image = hand_remover.process(processed_image, is_cropped=is_cropped)
        # print("Hand removal time: {}".format(time.time() - start_time))
        
        # post processing
        start_time = time.time()
        processed_image = stroke_filter.process(processed_image)
        # print("Post processing time: {}".format(time.time() - start_time))
            
        if OUTPUT_SIMULATED_CAMERA:
            if is_cropped:
                camera_frame = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                camera.schedule_frame(camera_frame)
        

def connect_clicked():
    global new_camera_url
    new_camera_url = window.cameraUrlInput.toPlainText()

# setup app
app = QtWidgets.QApplication(sys.argv)
window = uic.loadUi("libs/main_window.ui")
window.setWindowTitle("PaperStream app")
window.newRoomBtn.clicked.connect(new_room_clicked)
window.joinRoomBtn.clicked.connect(join_room_clicked)
window.connectBtn.clicked.connect(connect_clicked)
window.cameraUrlInput.setPlainText(DEFAULT_WEBCAM_URL)
window.show()

_thread.start_new_thread( camera_reading_thread, () )
_thread.start_new_thread( processing_thread, () )
app.exec()
