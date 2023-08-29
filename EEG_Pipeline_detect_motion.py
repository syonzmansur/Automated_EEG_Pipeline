#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import EEG_Pipeline_functions as fz
import math
import sys
import ast

# global figure options
plt.rcParams["figure.figsize"] = (15,5)

input_dir = sys.argv[1]
print(input_dir)
output_dir = sys.argv[2]
print(output_dir)
batch_id = sys.argv[3]
print(batch_id)
current_channel = sys.argv[4]
print(current_channel)

def remove_after_underscore(input_string):
    if "_" in input_string:
        return input_string.split("_")[0]
    else:
        return input_string

current_channel_mouse_id = remove_after_underscore(current_channel)

def remove_after_period(input_string):
    if "." in input_string:
        return input_string.split(".")[0]
    else:
        return input_string

current_channel_mouse_id = remove_after_period(current_channel)

# check if metadata file exists
metadata_file = input_dir + "/" + batch_id + "_metadata.csv"
if os.path.isfile(metadata_file):
    metadata = pd.read_csv(metadata_file)
    print("Metadata file exists.")
else:
    error = "Designated metadata file does not exist! exiting..."
    sys.exit(error)

os.makedirs(f'{output_dir}/Motion_Tracking/Motion_Outputs/{current_channel_mouse_id}', exist_ok=True)

# Import frame rate
frame_pts = pd.read_csv(f'{output_dir}/Motion_Tracking/Motion_Intermediates/{batch_id}_frame_rate.txt', names = ["pts", "frame"])
pts = frame_pts['pts']*1000
pts = pts.astype(int)

# Import segment data
segment_csv = output_dir + "/Motion_Tracking/Motion_Intermediates/" + batch_id + "_segment_data.csv"
segment_data = pd.read_csv(segment_csv)

# Set start & end frames
day1_start = segment_data.loc[segment_data['names'] == 'day1_start']['frame'].tolist()[0]
day1_end = segment_data.loc[segment_data['names'] == 'day1_end']['frame'].tolist()[0]
night_start = segment_data.loc[segment_data['names'] == 'night_start']['frame'].tolist()[0]
night_end = segment_data.loc[segment_data['names'] == 'night_end']['frame'].tolist()[0]
day2_start = segment_data.loc[segment_data['names'] == 'day2_start']['frame'].tolist()[0]
day2_end = segment_data.loc[segment_data['names'] == 'day2_end']['frame'].tolist()[0]

# Video data
if not math.isnan(day1_start):
    # Import day 1 video
    day1_video_filename = output_dir + "/Motion_Tracking/Motion_Intermediates/" + batch_id + "_" + current_channel_mouse_id + "_day1_minus_background.mp4"
    day1_cap = cv2.VideoCapture(day1_video_filename)
    day1_n_frames = round(day1_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = day1_cap.read()
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = np.shape(frame)
    
    # Identify optimal motion threshold
    day1_motion, day1_mt_cutoff, day1_exclude_frames = fz.find_motion_threshold(day1_video_filename, day1_n_frames, shape)
    print("Day 1 motion cutoff: ", day1_mt_cutoff)
    
    # Find mean frame rate
    day1_mean_frame_rate = pts[len(day1_motion)-1]/(len(day1_motion))
    print("Day 1 mean frame rate: ", day1_mean_frame_rate)
    
if not math.isnan(day2_start):
    # Import day 2 video
    day2_video_filename = output_dir + "/Motion_Tracking/Motion_Intermediates/" + batch_id + "_" + current_channel_mouse_id + "_day2_minus_background.mp4"
    day2_cap = cv2.VideoCapture(day2_video_filename)
    day2_n_frames = round(day2_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = day1_cap.read()
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = np.shape(frame)
    
    # Identify optimal motion threshold
    day2_motion, day2_mt_cutoff, day2_exclude_frames = fz.find_motion_threshold(day2_video_filename, day2_n_frames, shape)
    print("Day 2 motion cutoff: ",day2_mt_cutoff)
    
    # Find mean frame rate
    day2_mean_frame_rate = pts[len(day2_motion)-1]/(len(day2_motion))
    print("Day 2 mean frame rate: ", day2_mean_frame_rate)
else:
    day2_mt_cutoff = 0
    day2_exclude_frames = 0

### CROP TO START & END INDEX? ###
# Convert motion per frame to motion per millisecond
motion_ms = np.repeat(np.nan,pts[0]).tolist()
frame_ms = np.repeat(frame_pts['frame'][0],pts[0]).tolist()
print(len(motion_ms))
print(len(frame_ms))
# Deal w/ missing values
if math.isnan(day1_start):
    day1_start = 0
    day1_end = 0
    day1_exclude_frames = []
if math.isnan(night_start):
    print(night_start)
    night_start = 0
    night_end = 0
if math.isnan(day2_start):
    print(day2_start)
    day2_start = 0
    day2_end = 0
    day2_exclude_frames = []
    
day1_start = int(day1_start)
day1_end = int(day1_end)
night_start = int(night_start)
night_end = int(night_end)
day2_start = int(day2_start)
day2_end = int(day2_end)

for i in tqdm(range(1,len(pts))):
    # Frame
    frame_ms.extend(np.repeat(frame_pts['frame'][i],(pts[i]-pts[i-1])).tolist())
    # Motion depends on time period
    if day1_start <= i <= (day1_end - 1): # Day 1
        if i in day1_exclude_frames:
            motion_ms.extend(np.repeat(np.nan,(pts[i]-pts[i-1])).tolist())
        else:
            motion_ms.extend(np.repeat(day1_motion[(i-(day1_start-1))],(pts[i]-pts[i-1])).tolist())
    elif night_start <= i <= night_end: # Night
        motion_ms.extend(np.repeat(np.nan,(pts[i]-pts[i-1])).tolist())
    elif day2_start <= i <= day2_end: # Day 2
        if i in day2_exclude_frames_shifted:
            motion_ms.extend(np.repeat(np.nan,(pts[i]-pts[i-1])).tolist())
        else:
                motion_ms.extend(np.repeat(day2_motion[(i-(day2_start-1))],(pts[i]-pts[i-1])).tolist())
    else:
        motion_ms.extend(np.repeat(np.nan,(pts[i]-pts[i-1])).tolist())    

# Detect periods of inactivity
inactive_threshold = 12
min_duration = 1000 #(milliseconds)
inactive_ms = fz.measure_inactivity(np.array(motion_ms), inactive_threshold, min_duration)
inactive_ms = inactive_ms.astype('float')
# Correct inactivity with missing data
for i in tqdm(range(1,len(inactive_ms))):
    if np.isnan(motion_ms[i]):
        inactive_ms[i] = np.nan

# Create dataframe
dict_ms = {'frame': frame_ms, 'motion': motion_ms, 'inactive': inactive_ms}
df_ms = pd.DataFrame(data=dict_ms)

# Convert back to per frame (for preview video)
df_frame = df_ms.groupby(['frame']).mean()
day1_end_ms_test = [loc for loc, val in enumerate(df_ms['frame'].tolist()) if val == (night_start-1)]
if day1_end_ms_test:
    day1_end_ms = max(day1_end_ms_test)
else:
    day1_end_ms = 0

if not math.isnan(day2_start):
    day2_start_ms = df_ms['frame'].tolist().index(day2_start)
else:
    day2_start_ms = 0
    day2_mt_cutoff = 0

# Save threshold parameters
threshold_dict = {'Parameter': ['Pixel value change', 'Number of pixels', 'Minimum inactivity duration (ms)', 'Start (ms)', 'End (ms)'], 'Day1_value': [day1_mt_cutoff, inactive_threshold, min_duration, 0, day1_end_ms], 'Day2_value': [day2_mt_cutoff, inactive_threshold, min_duration, day2_start_ms, len(motion_ms)]}         
threshold_df = pd.DataFrame(threshold_dict) 
threshold_df.to_csv(f'{output_dir}/Motion_Tracking/Motion_Outputs/{current_channel_mouse_id}/{current_channel_mouse_id}_motion_threshold_parameters.csv', index=False)

# Find duration of inactive periods
inactivity_duration = []
count = 0
old = inactive_ms[0]
if (np.isnan(old)):
    inactivity_duration.extend([np.nan])
elif (old == 0):
    inactivity_duration.extend([0])
elif (old == 100):
    count = count + 1
# Loop through
for i in tqdm(range(1, len(inactive_ms))):
    new = inactive_ms[i]
    if (old == 100):
        if (np.isnan(new)):
            inactivity_duration.extend([count]*count)
            inactivity_duration.extend([np.nan])
            count = 0
        elif (new == 0):
            inactivity_duration.extend([count]*count)
            inactivity_duration.extend([0])
            count = 0
        elif (new == 100):
            count = count + 1
    else:
        if (np.isnan(new)):
            inactivity_duration.extend([np.nan])
        elif (new == 0):
            inactivity_duration.extend([0])
        elif (new == 100):
            count = count + 1
    if ((i == (len(inactive_ms)-1)) & (new == 100)):
        inactivity_duration.extend([count]*count)
    old = new

# Find duration of active periods
activity_duration = []
count = 0
old = inactive_ms[0]
if (np.isnan(old)):
    activity_duration.extend([np.nan])
elif (old == 100):
    activity_duration.extend([0])
elif (old == 0):
    count = count + 1
# Loop through
for i in tqdm(range(1, len(inactive_ms))):
    new = inactive_ms[i]
    if (old == 0):
        if (np.isnan(new)):
            activity_duration.extend([count]*count)
            activity_duration.extend([np.nan])
            count = 0
        elif (new == 100):
            activity_duration.extend([count]*count)
            activity_duration.extend([0])
            count = 0
        elif (new == 0):
            count = count + 1
    else:
        if (np.isnan(new)):
            activity_duration.extend([np.nan])
        elif (new == 100):
            activity_duration.extend([0])
        elif (new == 0):
            count = count + 1
    if ((i == (len(inactive_ms)-1)) & (new == 0)):
        activity_duration.extend([count]*count)
    old = new

# Add to dataframe
df_ms['inactive_duration'] = inactivity_duration
df_ms['active_duration'] = activity_duration
# Save
df_ms.to_csv(f'{output_dir}/Motion_Tracking/Motion_Outputs/{current_channel_mouse_id}/{current_channel_mouse_id}_motion.csv')

# Find start & stop points for preview video

# Reset start & end frames
day1_start = segment_data.loc[segment_data['names'] == 'day1_start']['frame'].tolist()[0]
day1_end = segment_data.loc[segment_data['names'] == 'day1_end']['frame'].tolist()[0]
day2_start = segment_data.loc[segment_data['names'] == 'day2_start']['frame'].tolist()[0]
day2_end = segment_data.loc[segment_data['names'] == 'day2_end']['frame'].tolist()[0]

# Day 1
if not math.isnan(day1_start):
    day1_prev_start = 20000
    done = False
    while ((day1_prev_start < (len(day1_motion) - 30000)) & (done == False)):
        day1_prev_end = day1_prev_start + 10000
        count_active = df_frame['inactive'][day1_prev_start:day1_prev_end] == 0
        percent_active = count_active.sum()/10000
        print(percent_active)
        if (0.025 < percent_active < 0.90):
            done = True
        else:
            day1_prev_start = day1_prev_start + 10000
    day1_prev_end = day1_prev_start + 10000
    print(day1_prev_start)
    print(day1_prev_end)
    fz.write_preview_video(current_channel_mouse_id,
                           output_dir,
                           df_frame['inactive'],
                           day1_video_filename,
                           day1_mt_cutoff,
                           day1_mean_frame_rate,
                           shape, 
                           day1_prev_start,
                           day1_prev_end)

# Day 2
if not math.isnan(day2_start):
    day2_prev_start = 20000
    done = False
    while ((day2_prev_start < (len(day2_motion) - 30000)) & (done == False)):
        day2_prev_end = day2_prev_start + 10000
        count_active = df_frame['inactive'][(day2_frame + day2_prev_start):(day2_frame + day2_prev_end)] == 0
        percent_active = count_active.sum()/10000
        print(percent_active)
        if (0.025 < percent_active < 0.90):
            done = True
        else:
            day2_prev_start = day2_prev_start + 10000
    day2_prev_end = day2_prev_start + 10000
    print(day2_prev_start)
    print(day2_prev_end)

    fz.write_preview_video(current_channel_mouse_id,
                           output_dir,
                           df_frame['inactive'],
                           day2_video_filename,
                           day2_mt_cutoff,
                           day2_mean_frame_rate,
                           shape, 
                           day2_frame,
                           day2_prev_start,
                           day2_prev_end)

# Plot motion (all)
plt.plot(np.arange(len(motion_ms)), motion_ms)
plt.xlabel("Time (milliseconds)")
plt.ylabel("Motion score (# of pixels over motion threshold)")
plt.savefig(f'{output_dir}/Motion_Tracking/Motion_Outputs/{current_channel_mouse_id}/{current_channel_mouse_id}__motion_plot.png')

plt.clf()
mini_motion = []
for value in motion_ms:
    if not (np.isnan(value) or value == 0):
        mini_motion.append(value)
print(mini_motion)

plt.plot(np.arange(len(mini_motion[10000:20000])), mini_motion[10000:20000])
plt.xlabel("Time (milliseconds)")
plt.ylabel("Motion score (# of pixels over motion threshold)")
plt.title("Mini Motion Plot")
plt.savefig(f'{output_dir}/Motion_Tracking/Motion_Outputs/{current_channel_mouse_id}/{current_channel_mouse_id}__motion_plot_mini.png')
plt.show()
