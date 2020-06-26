import numpy as np
import cv2
import os
import time
from distutils import util
import argparse
import time
import urllib 
from libs.paper_processor import PaperProcessor

paper_processor = PaperProcessor(smooth=False, debug=True)

# load video file
cap = cv2.VideoCapture("http://192.168.43.1:8080/video")

while(True):
    # read next frame from VideoCapture
    ret, frame = cap.read()
    if frame is not None:

        ret, warped_image = paper_processor.transform_image(frame)
        
        cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)
        cv2.imshow("Debug",  warped_image)
        
    # exit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and video output
if args.outputVideo is not None:
    videoOut.release()
cap.release()
# close cv2 window
cv2.destroyAllWindows()