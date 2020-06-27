from sklearn.cluster import KMeans
from collections import Counter
import cv2
import numpy as np
from threading import Thread, Lock
import time


class HandRemover(object):

    def __init__(self):
        self.lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
        self.upper_HSV_values = np.array([25, 255, 255], dtype="uint8")

        self.lower_YCbCr_values = np.array((0, 138, 67), dtype = "uint8")
        self.upper_YCbCr_values = np.array((255, 173, 133), dtype = "uint8")

        self.background = None
        
    def process(self, image, is_cropped):
        if is_cropped == False:
            return image

        if self.background is None:
            self.background = image
        
        hand_mask = self.__get_hand_mask(image)
        
        background_area = np.where(hand_mask==0)
        self.background[background_area] = image[background_area]
        
        # self.background = self._enhance_image(self.background)
        
        # self.background[]

        return self.background

    def __get_hand_mask(self, image):
        HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        YCbCr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

        mask_YCbCr = cv2.inRange(YCbCr_image, self.lower_YCbCr_values, self.upper_YCbCr_values)
        mask_HSV = cv2.inRange(HSV_image, self.lower_HSV_values, self.upper_HSV_values)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))

        foreground_mask = cv2.add(mask_HSV, mask_YCbCr)

        # Morphological operations

        background_mask = ~foreground_mask
        background_mask = cv2.erode(background_mask, kernel, iterations=50)
        background_mask[background_mask==255] = 128

        marker = cv2.add(foreground_mask, background_mask)
        marker = np.int32(marker)
        cv2.watershed(image, marker)

        m = cv2.convertScaleAbs(marker)
        m[m != 255] = 0
        m = m.astype(np.uint8)

        m = cv2.dilate(m, kernel, iterations=50)
        # cv2.imshow('m', m)
        
        return m
        
    def __enhance_image(self, image):
        rgb_planes = cv2.split(image)

        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

        result = cv2.merge(result_planes)
        result_norm = cv2.merge(result_norm_planes)
        
        return result_norm