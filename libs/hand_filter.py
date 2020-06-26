import cv2
import numpy as np 


class SkinColorFilter(object):
    
    def __init__(self):
        self.lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
        self.upper_HSV_values = np.array([25, 255, 255], dtype="uint8")

        self.lower_YCbCr_values = np.array((0, 138, 67), dtype = "uint8")
        self.upper_YCbCr_values = np.array((255, 173, 133), dtype = "uint8")

        self.binary_threshold = [0, 128]

    def run(self, image):
        binary_mask_image = self.__color_segmentation(image)
        output = self.__region_based_segmentation(image, binary_mask_image)

        return output


    def __color_segmentation(self, image):
        HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        YCbCr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        
        # A binary mask is returned. White pixels (255) represent pixels that fall into the upper/lower.
        mask_YCbCr = cv2.inRange(YCbCr_image, self.lower_YCbCr_values, self.upper_YCbCr_values)
        mask_HSV = cv2.inRange(HSV_image, self.lower_HSV_values, self.upper_HSV_values)
        
        binary_mask_image = cv2.add(mask_HSV,mask_YCbCr)

        return binary_mask_image

    # Function that applies Watershed and morphological operations on the thresholded image
    def __region_based_segmentation(self, image, binary_mask_image):
        # Morphological operations
        image_foreground = cv2.erode(binary_mask_image, None, iterations=3) # Remove noise
        dilated_binary_image = cv2.dilate(binary_mask_image, None, iterations=3) # The background region is reduced a little because of the dilate operation
        ret, image_background = cv2.threshold(dilated_binary_image, self.binary_threshold[0], self.binary_threshold[1], cv2.THRESH_BINARY) # set all background regions to 128

        # Add both foreground and background, forming markers. The markers are "seeds" of the future image regions.
        image_marker = cv2.add(image_foreground, image_background)
        # Convert to 32SC1 format
        image_marker32 = np.int32(image_marker)

        cv2.watershed(image, image_marker32)
        # Convert back to uint8
        m = cv2.convertScaleAbs(image_marker32)

        # bitwise of the mask with the input image
        ret, background_mask = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        output = cv2.bitwise_and(image, image, mask = ~background_mask)
        
        return output