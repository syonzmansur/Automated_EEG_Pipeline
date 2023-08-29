#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Functions for EEG video motion detection pipeline

# Get average pixel value of each frame to assess change from day to night
# Extract frames for background calculation
def get_pixel_value_and_extract_frames(video_filename, start_frame, end_frame, downsample_factor):
    # import video
    vid_in = cv2.VideoCapture(video_filename)
    # set up lists
    avg_pixel_values = []
    frame_ids = []
    frames = []
    # iterate through frames with progress bar
    for i in tqdm(range(1,end_frame+1)):
        success,image = vid_in.read()
        if success == True and i >= start_frame:
            # get average pixel value of each frame
            avg_pixel_values.extend([np.average(image)])
            frame_ids.extend([i])
            # for selected interval, get frame for background calculation
            if i%downsample_factor==0:
                frames.append(image)
    # return average pixel values and background image
    return avg_pixel_values, frame_ids, frames


# Write preview video
def write_preview_video(mouse, output_dir, inactive_frame, day1_video_filename, day1_mt_cutoff,
                       fps, shape, day1_prev_start, day1_prev_end):
   
    # Day 1 input video
    day1_vid_in = cv2.VideoCapture(day1_video_filename)
    day1_vid_in.set(cv2.CAP_PROP_POS_FRAMES, day1_prev_start) 
    
    width = shape[1]
    height = shape[0]*2
    
    # list for output video
    vid_out = cv2.VideoWriter(f'{output_dir}/final_output/{mouse}_preview_video.mp4',cv2.VideoWriter_fourcc(*'FMP4'), fps, (width, height))
        
    # set text parameters
    textfont = cv2.FONT_HERSHEY_SIMPLEX
    textposition = (3,8)
    textposition2 = (3,20)
    textfontscale = 0.255
    textfontscale2 = 0.4
    textlinetype = 1
    textfontcolor = 255
    
    ## Day 1

    #Initialize first frame
    ret, frame_new = day1_vid_in.read()
    frame_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
    frame_new = cv2.GaussianBlur(frame_new.astype('float'),(0,0),1)
    
    #Loop through frames to detect frame by frame differences
    for x in tqdm(range(day1_prev_start+1,day1_prev_end)):

        #Attempt to load next frame
        frame_old = frame_new
        ret, frame_new = day1_vid_in.read()
        if ret == True:
            
            #process frame           
            frame_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
            frame_new = cv2.GaussianBlur(frame_new.astype('float'),(0,0),1) 
            frame_dif = np.absolute(frame_new - frame_old)
            frame_cut = (frame_dif > day1_mt_cutoff).astype('uint8')*255
            frame_cut_color = cv2.cvtColor(frame_cut, cv2.COLOR_GRAY2BGR)

            #Add text to videos, display and save
            text1 = "DAY 1:"
            texttext = 'INACTIVE' if inactive_frame[x]==100 else 'ACTIVE'
            frame_new_text = cv2.cvtColor(frame_new.astype('uint8'), cv2.COLOR_GRAY2BGR)
            cv2.putText(frame_new_text,text1,textposition,textfont,textfontscale,textfontcolor,textlinetype)
            cv2.putText(frame_new_text,texttext,textposition2,textfont,textfontscale2,textfontcolor,textlinetype)
            display = np.concatenate((frame_new_text.astype('uint8'),frame_cut_color))
            
            vid_out.write(display)
    
    vid_out.release()

# Measure motion (adapted from ezTrack, Pennington et al. 2019)
def measure_motion (input_video_filename, n_frames, mt_cutoff, SIGMA=1):
    """ 
    -------------------------------------------------------------------------------------
    
    Loops through segment of video file, frame by frame, and calculates number of pixels 
    per frame whose intensity value changed from prior frame.
    
    -------------------------------------------------------------------------------------
    Args:
        input_video_filename
        
        n_frames
                
        mt_cutoff:: [float]
            Threshold value for determining magnitude of change sufficient to mark
            pixel as changing from prior frame.
                
        SIGMA:: [float]
            Sigma value for gaussian filter applied to each image. Passed to 
            OpenCV `cv2.GuassianBlur`.
    
    -------------------------------------------------------------------------------------
    Returns:
        Motion:: [numpy.array]
            Array containing number of pixels per frame whose intensity change from
            previous frame exceeds `mt_cutoff`. Length is number of frames passed to
            function to loop through. Value of first index, corresponding to first frame,
            is set to 0.
    
    -------------------------------------------------------------------------------------
    Notes:

    """
    
    #Upload file
    cap = cv2.VideoCapture(input_video_filename)
    cap_max = n_frames

    #Initialize first frame and array to store motion values in
    ret, frame_new = cap.read()
    frame_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
    frame_new = cv2.GaussianBlur(frame_new.astype('float'),(0,0),SIGMA)  
    Motion = np.zeros(cap_max)

    #Loop through frames to detect frame by frame differences
    for x in tqdm(range(1,len(Motion))):
        frame_old = frame_new
        ret, frame_new = cap.read()
        if ret == True:
            #Reset new frame and process calculate difference between old/new frames
            frame_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
            frame_new = cv2.GaussianBlur(frame_new.astype('float'),(0,0),SIGMA)  
            frame_dif = np.absolute(frame_new - frame_old)
            frame_cut = (frame_dif > mt_cutoff).astype('uint8')
            Motion[x]=np.sum(frame_cut)
        else: 
            #if no frame is detected
            x = x-1 #Reset x to last frame detected
            Motion = Motion[:x] #Amend length of motion vector
            break
        
    cap.release() #release video
    print('total frames processed: {f}\n'.format(f=len(Motion)))
    return(Motion) #return motion values

# Find optimal motion threshold
def find_motion_threshold(input_video_filename, n_frames, shape):
    i = 5
    done = False
    done_final = False
    # Hop by 5s
    while (i < 100) & (done == False):
        # Measure motion using cutoff
        Motion = measure_motion(input_video_filename, n_frames, i, SIGMA=1)
        print(i)
        # Find outliers due to camera jostling outside of first 5 minutes
        if (i == 5):
            high_motion_frames = np.where(Motion >= (shape[0]*shape[1])*0.5)[0].tolist()
            exclude_frames = np.arange(18000).tolist()
            for h in high_motion_frames:
                # Find 2 min +/-
                if (h > 7200):
                    min_h = h - 7200
                else:
                    min_h = 0
                if (h < (len(Motion)-7200)):
                    max_h = h + 7200
                else:
                    max_h = len(Motion)
                # Create range and add to list
                exclude_h = np.arange(min_h, max_h).tolist()
                exclude_frames.extend(exclude_h)
                exclude_frames = list(set(exclude_frames))
        # Filter motion
        motion_filtered = np.delete(Motion, exclude_frames)
        # On average, how many pixels moved in the 5 frames with most motion?
        # As a percent of the whole?
        avg_top5 = np.mean(-np.sort(-np.array(motion_filtered))[0:5])
        avg_top5_percent = avg_top5/(shape[0]*shape[1])
        print(avg_top5_percent)
        
        # How many pixels moved in the frame with 5th-most motion?
        # As a percent of the whole?
        num_5 = -np.sort(-np.array(motion_filtered))[5]
        num_5_percent = num_5/(shape[0]*shape[1])
        print(num_5_percent)
    
        # How many frames had zero movement?
        count_zero = motion_filtered == 0
        percent_zero = count_zero.sum()/len(motion_filtered)
        print(percent_zero)
    
        if (avg_top5_percent < 0.08) & (num_5_percent < 0.07) & (percent_zero > 0.5):
            done = True
        else:
            i = i + 5

    # Now go back by 5 and hop by 1
    j = i - 4
    while (j <= i) & (done_final == False):
        # Measure motion using cutoff
        Motion = measure_motion(input_video_filename, n_frames, j, SIGMA=1)
        print(j)
        # Filter motion
        motion_filtered = np.delete(Motion, exclude_frames)
        # On average, how many pixels moved in the 5 frames with most motion?
        # As a percent of the whole?
        avg_top5 = np.mean(-np.sort(-np.array(motion_filtered))[0:5])
        avg_top5_percent = avg_top5/(shape[0]*shape[1])
        print(avg_top5_percent)
        
        # How many pixels moved in the frame with 5th-most motion?
        # As a percent of the whole?
        num_5 = -np.sort(-np.array(motion_filtered))[5]
        num_5_percent = num_5/(shape[0]*shape[1])
        print(num_5_percent)
    
        # How many frames had zero movement?
        count_zero = motion_filtered == 0
        percent_zero = count_zero.sum()/len(motion_filtered)
        print(percent_zero)
    
        if (avg_top5_percent < 0.08) & (num_5_percent < 0.07) & (percent_zero > 0.5):
            done_final = True
        else:
            j = j + 1
    
    # Final value is +1
    j = j + 1
    Motion = measure_motion(input_video_filename, n_frames, j, SIGMA=1)
    # Filter motion
    motion_filtered = np.delete(Motion, exclude_frames)
    # On average, how many pixels moved in the 5 frames with most motion?
    # As a percent of the whole?
    avg_top5 = np.mean(-np.sort(-np.array(motion_filtered))[0:5])
    avg_top5_percent = avg_top5/(shape[0]*shape[1])
    print(avg_top5_percent)
        
    # How many pixels moved in the frame with 5th-most motion?
    # As a percent of the whole?
    num_5 = -np.sort(-np.array(motion_filtered))[5]
    num_5_percent = num_5/(shape[0]*shape[1])
    print(num_5_percent)
    
    # How many frames had zero movement?
    count_zero = motion_filtered == 0
    percent_zero = count_zero.sum()/len(motion_filtered)
    print(percent_zero)
    mt_cutoff = j
    return(Motion, mt_cutoff, exclude_frames)

# Measure periods of inactivity (adapted from ezTrack, Pennington et al. 2019)
def measure_inactivity(Motion, Inactivity_Threshold, MinDuration = 0):
    """ 
    -------------------------------------------------------------------------------------
    
    Calculates freezing on a frame by frame basis based upon measure of motion.

    """

    #Find frames below thresh
    BelowThresh = (Motion < Inactivity_Threshold).astype(int)

    #Perform local cumulative thresh detection
    #For each consecutive frame motion is below threshold count is increased by 1 until motion goes above thresh, 
    #at which point coint is set back to 0
    CumThresh = np.zeros(len(Motion))
    for x in range (1,len(Motion)):
        if (BelowThresh[x]==1):
            CumThresh[x] = CumThresh[x-1] + BelowThresh[x]

    #Define periods where motion is below thresh for minduration as freezing
    Freezing = (CumThresh>=MinDuration).astype(int)
    for x in range( len(Freezing) - 2, -1, -1) : 
        if Freezing[x] == 0 and Freezing[x+1]>0 and Freezing[x+1]<MinDuration:
            Freezing[x] = Freezing[x+1] + 1
    Freezing = (Freezing>0).astype(int)
    Freezing = Freezing*100 #Convert to Percentage
    
    return(Freezing)


