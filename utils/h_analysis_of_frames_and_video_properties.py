#######################################################################################
# contains all the functions needed to analyze the video frames to produce a data frame
#######################################################################################
import numpy as np
import pandas as pd
import scipy

import cv2
import pdb

#----------------------------------------------------------------------------------#
#---------------------- video properties ------------------------------------------#
#----------------------------------------------------------------------------------#
def print_data_of_interest_from_cap(caps, vid, new_GoProVideo_flag):
    frame_width = caps.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = caps.get(cv2.CAP_PROP_FRAME_HEIGHT)
    framerate = caps.get(cv2.CAP_PROP_FPS)
    num_frames = caps.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_vid = num_frames/framerate
    print("------- video properties of interest -------")
    print("Loaded Video File: ", vid)
    print("Frame width: ", frame_width)
    print("Frame height: ", frame_height)
    print("Frame rate (openCV DEF): ",framerate )
    print("#Frames in video file: ",num_frames )
    print("Duration = frames/FPS: ", duration_vid)
    print("--------- end video properties of interest -------\n")

def set_and_get_frame_of_interest(frame_num,video):
    '''
    Only use for unimportant information -> cap.set does not operate safely and as expected!
    -> For our plot though, it is fine. But for publication, choose a frame I know, for example the first frame where I detect beam on?

    setting the frame_num does not seam to be a save approach. go via FPS and MSEC
    returns:    the frame at desired frame number but also sets cap to it
    '''
    caps = cv2.VideoCapture(video)
    frame_time_in_msec = 1000*frame_num/(caps.get(cv2.CAP_PROP_FPS))
    caps.set(cv2.CAP_PROP_POS_MSEC, frame_time_in_msec)
    stat, frame_of_interest = caps.read()
    caps.release()
    return frame_of_interest

#----------------------------------------------------------------------------------#
#----------------------------- frame analysis -------------------------------------#
#----------------------------------------------------------------------------------#

#FRAME AND ROI PROPERTIES
def analyze_average_roi(img, roi):
    cropped_img = img[roi[1]:roi[3], roi[0]:roi[2]] 
    av_roi = np.average(cropped_img)
    return av_roi

#SURFACE
def analyze_px_roi_surf(img, threshold_raw, threshold_normalized, roi, mode): # my current px analysis technique (any)
    #TODO: add upper boundary
    '''roi: only analyze masked region, so crop image for analysis'''
    #my input for development
    normalized = False
    fill_holes = False

    if normalized:
        img = img/np.average(img)
        threshold = threshold_normalized
    else:
        threshold = threshold_raw

    cropped_img = img[roi[1]:roi[3], roi[0]:roi[2]] 
    #watch out for coordinate systems:
    # opencv: x from left to right, y from top to bottom, 
    # # numpy: x from top to bottom, y from left to right
    # opencv rectangle: roi = (220,20,380,90)=(upperleft_x, upperleft_y, lowerright_x, lowerright_y)--> numpy image region: [20:90, 220:380]
    
    #pdb.set_trace()
    if mode == "BGR" or mode =="RGB":
        val_pxs = np.sum(np.where(((cropped_img > threshold[0]) &(cropped_img < threshold[1])), 255,0))/255 #cnts px > threshold# counts number of pixels exceeding the threshold
    elif mode == "HSV":
        val_pxs = np.sum(np.where(((cropped_img > threshold[0]) &(cropped_img < threshold[1])),1,0)) #TODO: kann ich das nicht überall auf 1 setzen?
    
    # break down:
    # - np.where(cropped_img > threshold, 255. 0): if condition True, then set value to 255, else 0
    # - np.sum( ...)/255: sums up all the pixel values where condition satisfied. As these are only 255 and we want he number of pixels,
    #   this is normalized, so pixels values are 1
    # 
    # this works like a mask!

    #similar to above but now with filled holes
    if fill_holes:
        #TODO
        pass

    return val_pxs, np.size(cropped_img)

# SCINTILLATOR
def analyze_px_roi_scinti(img, threshold, roi, mode):
    #TODO compare statistics and smoothness of curve what is better, avg, sum, median etc.
    cropped_img=img[roi[1]:roi[3], roi[0]:roi[2]] #coordinate system translation opencv -> numpy: x->y, y->x
    if mode != "HSV":
        #val_pxs = np.median(cropped_img)
        #val_pxs = np.average(cropped_img > threshold)
        #val_pxs = np.average(cropped_img)/255
        val_pxs = np.average(cropped_img[(cropped_img > threshold[0]) & (cropped_img < threshold[1])])
        #TODO: is array flattened after this??

        #val_pxs = np.sum(np.where(cropped_img > threshold, 255,0))/255
    else:
        val_pxs = np.sum(np.where(((cropped_img > threshold[0]) &(cropped_img < threshold[1])),1,0))
    return val_pxs, np.size(cropped_img)

# LED
def find_center_of_mass_LED(channelled_img, threshold, roi):
    ''' split the img in blue, green, red channel; extract only red channel,  '''
    r_thresholded = np.where(((channelled_img > threshold[0])&(channelled_img<threshold[1])),255,0)
    r_thresholded_cropped = np.copy(r_thresholded)[roi[1]:roi[3], roi[0]:roi[2]] #+-1 to not include rectangle markers
    com_red = scipy.ndimage.center_of_mass(r_thresholded_cropped) # x and y vice versa!!

    return com_red

def find_center_LED(channelled_img, threshold, roi):
    r_thresholded = np.where(((channelled_img > threshold[0])&(channelled_img<threshold[1])),255,0)
    r_thresholded_cropped = np.copy(r_thresholded)[roi[1]:roi[3], roi[0]:roi[2]] #+-1 to not include rectangle markers
    com_red = scipy.ndimage.center_of_mass(r_thresholded_cropped) # x and y vice versa!!

    return com_red
    