import numpy as np
import cv2
import os
import time
from distutils import util
import argparse
import time
import urllib 
from filter_color import skinColorFilter

color_filter = skinColorFilter()

parser = argparse.ArgumentParser(description = 'Marker Tracking & Pose Estimation')
parser.add_argument('--inputVideo', type = str, help = 'Path to the video of the object to be tracked')
parser.add_argument('--referenceImage', type = str, help = 'Path to an image of the object to track including markers')
parser.add_argument('--outputVideo', type = str, default = None, help = 'Optional - Path to output video')
parser.add_argument('--smooth', type = util.strtobool, default = False, help = 'Should smooth transformation matrix?')
args = parser.parse_args()

# a little helper function for getting all dettected marker ids
# from the reference image markers
def which(x, values):
    indices = []
    for ii in list(values):
        if ii in x:
            indices.append(list(x).index(ii))
    return indices

# load video file
cap = cv2.VideoCapture("http://192.168.43.1:8080/video")
# open video file for writing
if args.outputVideo is not None:
    videoOut = cv2.VideoWriter(args.outputVideo, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(5)), (int(cap.get(3)), int(cap.get(4))))

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

# adjust dictionary parameters for better marker detection
parameters =  cv2.aruco.DetectorParameters_create()
parameters.cornerRefinementMethod = 5
parameters.errorCorrectionRate = 0.3

args.referenceImage = "ref.png"

# load reference image
refImage = cv2.cvtColor(cv2.imread(args.referenceImage), cv2.COLOR_BGR2GRAY)
# detect markers in reference image
refCorners, refIds, refRejected = cv2.aruco.detectMarkers(refImage, aruco_dict, parameters = parameters)
# create bounding box from reference image dimensions
rect = np.array([[[0,0],
                  [refImage.shape[1],0],
                  [refImage.shape[1],refImage.shape[0]],
                  [0,refImage.shape[0]]]], dtype = "float32")

if args.smooth:
    # for simple noise reduction we use deque
    from collections import deque
    # simple noise reduction
    h_array = deque(maxlen = 5)

def enhance_image(bgr):

    

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

    # bgr = color_filter.run(bgr)

    return bgr


last_transform_update = time.time()
M = None
M_inv = None

# cap = cv2.VideoCapture()

while(True):
    # read next frame from VideoCapture
    ret, frame = cap.read()
    if frame is not None:

        # convert frame to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect aruco markers in gray frame
        res_corners, res_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
        # if markers were detected
        if res_ids is not None:
            # find which markers in frame match those in reference image
            idx = which(refIds, res_ids)
            # if any detected marker in frame is also in the reference image
            if len(idx) > 0:
                # flatten the array of corners in the frame and reference image
                these_res_corners = np.concatenate(res_corners, axis = 1)
                these_ref_corners = np.concatenate([refCorners[x] for x in idx], axis = 1)
                # estimate homography matrix
                h, s = cv2.findHomography(these_ref_corners, these_res_corners, cv2.RANSAC, 10.0)
                # if we want smoothing
                if args.smooth:
                    h_array.append(h)
                    M = np.mean(h_array, axis = 0)
                else:
                    M = h

                # transform the rectangle using the homography matrix
                newRect = cv2.perspectiveTransform(rect, M, (gray.shape[1],gray.shape[0]))

                # draw the rectangle on the frame
                draw_frame = frame.copy()

                if M_inv is None:
                    print(time.time() - last_transform_update)
                    # estimate homography matrix
                    M_inv, s = cv2.findHomography(these_res_corners, these_ref_corners, cv2.RANSAC, 10.0)
                    last_transform_update = time.time()

                for r in res_corners:
                    nds = np.int32(r)
                    cv2.fillPoly(frame, nds, [255,255,255])

                frame_warp = cv2.warpPerspective(frame, M_inv, (refImage.shape[1], refImage.shape[0]))

                frame_warp = enhance_image(frame_warp)

                frame_warp = cv2.morphologyEx(frame_warp, cv2.MORPH_CLOSE, np.ones((1, 1), dtype=np.uint8))

            # draw detected markers in frame with their ids
            cv2.aruco.drawDetectedMarkers(draw_frame,res_corners,res_ids)
            
        
        else:
            draw_frame = None

        if draw_frame is None:
            draw_frame = frame
            frame_warp = frame
            # continue

        # if video is to be saved
        if args.outputVideo is not None:
            videoOut.write(draw_frame)
        # resize frame to half of both axes (because my screen is small!)
        draw_frame = cv2.resize(draw_frame, None, fx = 0.5, fy = 0.5)

        # Display the resulting frame
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.namedWindow('frame_warp', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', draw_frame)
        cv2.imshow('frame_warp', frame_warp)
        cv2.waitKey(1)
        
    # exit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and video output
if args.outputVideo is not None:
    videoOut.release()
cap.release()
# close cv2 window
cv2.destroyAllWindows()