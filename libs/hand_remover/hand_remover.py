from sklearn.cluster import KMeans
from collections import Counter
import cv2
import numpy as np
from threading import Thread, Lock
from skimage.filters import threshold_local
import time

class DominantColor(object):

    def __init__(self, name='thread dominant color'):
        self.image = None 
        self.dominant_color = None
        self.blank_paper = None
        self.stopped = False

        self.thread = Thread(target=self.update, name=name, args=())
        self.thread.daemon = True

    def update(self):
        while not self.stopped:
            if self.image is None:
                continue
            self.dominant_color, self.blank_paper = self.__update_dominant_color(self.image)
            # print(self.dominant_color)
            # break
            time.sleep(60)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped = True

    def update_image(self, image):
        self.image = image

    def get_color(self):
        return self.dominant_color

    def get_blank_paper(self):
        return self.blank_paper

    def __update_dominant_color(self, image, k=2):
        blank_paper = np.zeros_like(image).astype(np.uint8)
        #reshape the image to be a list of pixels
        image = cv2.resize(image, (128, 128))
        image = image.reshape((image.shape[0] * image.shape[1], 3))

        #cluster and assign labels to the pixels 
        clt = KMeans(n_clusters = k)
        labels = clt.fit_predict(image)

        #count labels to find most popular
        label_counts = Counter(labels)

        #subset out most popular centroid
        dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
        dominant_color = list(dominant_color)

        blank_paper[:] = dominant_color
        
        return dominant_color, blank_paper


class HandRemover(object):

    def __init__(self):
        self.lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
        self.upper_HSV_values = np.array([25, 255, 255], dtype="uint8")

        self.lower_YCbCr_values = np.array((0, 138, 67), dtype = "uint8")
        self.upper_YCbCr_values = np.array((255, 173, 133), dtype = "uint8")

        self.background = None

        self.dominant_color = DominantColor()
        self.dominant_color.start()
        
    def process(self, image, is_cropped):
        if is_cropped == False:
            return image

        self.dominant_color.update_image(image)

        if self.background is None:
            self.background = image
        
        hand_mask = self.__get_hand_mask(image)
        
        background_area = np.where(hand_mask==0)
        self.background[background_area] = image[background_area]
        
        # self.background = self._enhance_image(self.background)
        
        # self.background[]

        self.background = self.__flood_fill(self.background.copy())

        return self.background

    def __flood_fill(self, image):
        test = image
        test = cv2.Canny(test,250,300)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        test = cv2.dilate(test, kernel, iterations=3)
        
        # Copy the thresholded image.
        im_floodfill = test.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = test.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,int(w/2)), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = test | im_floodfill_inv

        test2 = image
        test2 = cv2.bitwise_and(test2, test2, mask=im_out)
        color = self.dominant_color.get_color()
        if color is not None:
            test2[test2[..., 0]==0] = color

        return test2

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