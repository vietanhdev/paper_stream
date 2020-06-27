import cv2
import numpy as np
from random import randint

class StrokeFilter:
    
    def __init__(self):
        
        self.prev_mask = None
        self.prev_connected_components = None
    
    def process(self, image):
        # image = self._post_processing(image)
        # image = self._get_stroke_mask(image)
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
        bgr[gray > 140] = [255, 255, 255]
        
        mask = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask[:,:] = 0
        mask[gray < 160] = 0

        bgr = cv2.morphologyEx(bgr, cv2.MORPH_CLOSE, np.ones((1, 1), dtype=np.uint8))
        
        # bgr[gray < 160] = [10, 10, 10]
        
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return bgr
    
    
    def _get_label_mask(self, labels, ids):

        mask = np.zeros(labels.shape[:2], dtype=np.uint8)
        for i in ids:
            mask[labels == i] = 255
        return mask
    
    def _get_stroke_mask(self, mask):
        
        nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if self.prev_connected_components is not None:
            prev_nb_components, prev_labels, prev_stats, prev_centroids = self.prev_connected_components
            
        if self.prev_mask is not None:
            
            new_diff = (mask & (~self.prev_mask)) | ((~mask) & self.prev_mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            new_diff = cv2.morphologyEx(new_diff, cv2.MORPH_OPEN, kernel)
            
            changed_labels = labels.copy()
            changed_labels[new_diff == 0] = 0
            changed_ids = np.unique(changed_labels).tolist()
            changed_ids.remove(0)
            
            new_label_mask = self._get_label_mask(labels, changed_ids)
            
            replaced_labels = prev_labels.copy()
            replaced_labels[new_label_mask == 0] = 0
            replaced_ids = np.unique(replaced_labels).tolist()
            replaced_ids.remove(0)
            
            replaced_labels_mask = self._get_label_mask(prev_labels, replaced_ids)
            
            combined_mask = self.prev_mask.copy()
            combined_mask[replaced_labels_mask > 0] = 0
            combined_mask[new_label_mask > 0] = 255
            
            mask = combined_mask
            
            nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
        cv2.namedWindow("Debug1", cv2.WINDOW_NORMAL)
        cv2.imshow("Debug1",  mask)
        cv2.waitKey(1)

            
        self.prev_mask = mask
        self.prev_connected_components = (nb_components, labels, stats, centroids)
            
        return mask