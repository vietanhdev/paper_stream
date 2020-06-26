import cv2
import numpy as np

from .hand_filter import *
from .stroke_filter import *
from .utils import *
from .image_processing import *
from .config import *

class PaperProcessor:

    def __init__(self, smooth=False, debug=False):

        self.smooth = smooth
        self.debug = debug

        # transform matrices
        self.M = None
        self.M_inv = None

        # define an empty custom dictionary with 
        aruco_dict = cv2.aruco.custom_dictionary(0, 4, 1)
        # add empty bytesList array to fill with 3 markers later
        aruco_dict.bytesList = np.empty(shape = (4, 2, 4), dtype = np.uint8)
        # add new marker(s)
        mybits = np.array([[1,0,1,1],[0,1,0,1],[0,0,1,1],[0,0,1,0]], dtype = np.uint8)
        aruco_dict.bytesList[0] = cv2.aruco.Dictionary_getByteListFromBits(mybits)
        mybits = np.array([[0,0,0,0],[1,1,1,1],[1,0,0,1],[1,0,1,0]], dtype = np.uint8)
        aruco_dict.bytesList[1] = cv2.aruco.Dictionary_getByteListFromBits(mybits)
        mybits = np.array([[1,0,0,1],[1,0,0,1],[0,1,0,0],[0,1,1,0]], dtype = np.uint8)
        aruco_dict.bytesList[2] = cv2.aruco.Dictionary_getByteListFromBits(mybits)
        mybits = np.array([[0,0,1,1],[0,0,1,1],[0,0,1,0],[1,1,0,1]], dtype = np.uint8)
        aruco_dict.bytesList[3] = cv2.aruco.Dictionary_getByteListFromBits(mybits)
        self.aruco_dict = aruco_dict

        # adjust dictionary parameters for better marker detection
        parameters =  cv2.aruco.DetectorParameters_create()
        parameters.cornerRefinementMethod = 5
        parameters.errorCorrectionRate = 0.3
        self.parameters = parameters

        # load reference image
        self.ref_image = cv2.cvtColor(cv2.imread(REFERENCE_ARUCO_IMAGE_PATH), cv2.COLOR_BGR2GRAY)

        # detect markers in reference image
        self.ref_corners, self.ref_ids, self.ref_rejected = cv2.aruco.detectMarkers(self.ref_image, aruco_dict, parameters = parameters)

        # create bounding box from reference image dimensions
        self.rect = np.array([[[0,0],
                        [self.ref_image.shape[1],0],
                        [self.ref_image.shape[1], self.ref_image.shape[0]],
                        [0,self.ref_image.shape[0]]]], dtype = "float32")


    def update_transform_matrices(self, gray):
        
        # detect aruco markers in gray frame
        res_corners, res_ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters = self.parameters)
        self.res_corners = res_corners
        self.res_ids = res_ids

        # if markers were not detected
        if res_ids is None:
            return False

        # find which markers in frame match those in reference image
        idx = which(self.ref_ids, res_ids)
        
        # if # of detected points is too small => ignore the result
        if len(idx) <= 2:
            return False

        # flatten the array of corners in the frame and reference image
        these_res_corners = np.concatenate(res_corners, axis = 1)
        these_ref_corners = np.concatenate([self.ref_corners[x] for x in idx], axis = 1)
        # estimate homography matrix
        h, s = cv2.findHomography(these_ref_corners, these_res_corners, cv2.RANSAC, 10.0)
        # if we want smoothing
        if self.smooth:
            h_array.append(h)
            self.M = np.mean(h_array, axis = 0)
        else:
            self.M = h

        # transform the rectangle using the homography matrix
        new_rect = cv2.perspectiveTransform(self.rect, self.M, (gray.shape[1],gray.shape[0]))

        self.M_inv, s = cv2.findHomography(these_res_corners, these_ref_corners, cv2.RANSAC, 10.0)

        return True


    def transform_image(self, image):

        # convert frame to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find new transform matrices
        is_aruco_detected = self.update_transform_matrices(gray)

        # draw detected markers in frame with their ids
        if self.debug and is_aruco_detected:
            draw_frame = image.copy()
            cv2.aruco.drawDetectedMarkers(draw_frame, self.res_corners, self.res_ids)
            cv2.namedWindow("Debug aruco", cv2.WINDOW_NORMAL)
            cv2.imshow("Debug aruco",  draw_frame)
            cv2.waitKey(1)
        
        # convert image using new transform matrices
        if is_aruco_detected:
            frame_warp = cv2.warpPerspective(image, self.M_inv, (self.ref_image.shape[1], self.ref_image.shape[0]))
            frame_warp = enhance_image(frame_warp)
            return True, frame_warp
        elif self.M_inv is not None:
            frame_warp = cv2.warpPerspective(image, self.M_inv, (self.ref_image.shape[1], self.ref_image.shape[0]))
            frame_warp = enhance_image(frame_warp)
            return True, frame_warp
        else:
            return False, image


    