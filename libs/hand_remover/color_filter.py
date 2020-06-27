import cv2
import numpy as np 


class SkinColorFilter(object):
    
    # class constructor
    def __init__(self, img):
        self.lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
        self.upper_HSV_values = np.array([25, 255, 255], dtype="uint8")

        self.lower_YCbCr_values = np.array((0, 138, 67), dtype = "uint8")
        self.upper_YCbCr_values = np.array((255, 173, 133), dtype = "uint8")

        self.background = img

    def run(self, image):
        binary_mask_image = self.__color_segmentation(image)
        background_mask = self.__region_based_segmentation(image, binary_mask_image)
        static_area = np.where(background_mask==0)
        self.background[static_area] = image[static_area]

        return self.background

    def __color_segmentation(self, image):
        HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        YCbCr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        
        # A binary mask is returned. White pixels (255) represent pixels that fall into the upper/lower.
        mask_YCbCr = cv2.inRange(YCbCr_image, self.lower_YCbCr_values, self.upper_YCbCr_values)
        mask_HSV = cv2.inRange(HSV_image, self.lower_HSV_values, self.upper_HSV_values)
        
        binary_mask_image = cv2.add(mask_HSV, mask_YCbCr)

        return binary_mask_image

    def __convert_to_convex_hull(self, background_mask):
        mask = np.zeros_like(background_mask).astype(np.uint8)

        contours, hierarchy = cv2.findContours(background_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

        # create hull array for convex hull points
        hull = []

        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], False))

        cv2.drawContours(mask, hull, -1, 255, -1)
        return mask

    # Function that applies Watershed and morphological operations on the thresholded image
    def __region_based_segmentation(self, image, foreground_mask):
        h, w = image.shape[:2]
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        
        background_mask = ~foreground_mask
        background_mask = cv2.erode(background_mask, kernel, iterations=60)
        background_mask[background_mask==255] = 128

        marker = cv2.add(foreground_mask, background_mask)
        # cv2.imshow('marker', marker)
        marker = np.int32(marker)

        cv2.watershed(image, marker)
        m = cv2.convertScaleAbs(marker)

        m[m != 255] = 0
        m = m.astype(np.uint8)

        m = cv2.dilate(m, kernel, iterations=20)
        
        return m