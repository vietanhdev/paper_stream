import cv2
import numpy as np

class StrokeFilter:
    
    def __init__(self):
        pass
    
    def process(self, image):
        """TODO: Update this"""
        return image
    
    def _post_processing(self, bgr):

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