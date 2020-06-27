import cv2
import numpy as np
from .hand_filter import *
from .stroke_filter import *

def post_process_image(bgr):

    gridsize = 11
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) 

    sigmaColor = 15
    sigmaSpace = 5
    bgr = cv2.bilateralFilter(bgr, -1, sigmaColor, sigmaSpace)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bgr[gray > 160] = [255, 255, 255]
    # bgr[gray < 160] = [0, 0, 0]

    bgr = cv2.morphologyEx(bgr, cv2.MORPH_CLOSE, np.ones((1, 1), dtype=np.uint8))

    return bgr