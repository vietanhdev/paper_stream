import numpy as np
import cv2
import os
import time
from distutils import util
import argparse
import time
import urllib 
from libs.paper_processor import PaperProcessor
from libs.pyfakewebcam import pyfakewebcam
from libs.filter_color import skinColorFilter
from libs.dominant_color import get_dominant_color_image


paper_processor = PaperProcessor(smooth=False, debug=True, output_video_path=None)
output_width, output_height = paper_processor.get_output_size()
# camera = pyfakewebcam.FakeWebcam('/dev/video3', output_width, output_height)

# load video file
cap = cv2.VideoCapture("http://192.168.43.1:8080/video")

dominant_color = None

while(True):
    # read next frame from VideoCapture
    ret, frame = cap.read()
    if frame is not None:
        ret, warped_image = paper_processor.transform_image(frame, enhance_image=False)

        if dominant_color is None:
            color_filter = skinColorFilter(warped_image)
            dominant_color_image, dominant_color = get_dominant_color_image(warped_image)

        background = color_filter.run(warped_image)

        gray_dominant_color_image = cv2.cvtColor(dominant_color_image, cv2.COLOR_BGR2GRAY)
        gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        diff = np.abs(gray_dominant_color_image.astype(np.int8) - gray_background.astype(np.int8))
        m = diff<30
        background[m] = 255
        # warped_image = background
        
        # if ret:
        #     warped_image_rgb = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
        #     camera.schedule_frame(warped_image_rgb)
        
        cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)
        cv2.imshow("Debug",  background)
        
    # exit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()