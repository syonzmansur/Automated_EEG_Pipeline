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
import EEG_Pipeline_functions as fz

# Arguments
input_dir = sys.argv[1]
print("Input directory: ", input_dir)
output_dir = sys.argv[2]
print("Output directory: ", output_dir)
batch_id = sys.argv[3]
print("Batch ID: ",batch_id)
video_file = sys.argv[4]
print("Input video file path: ", video_file)
metadata_file = sys.argv[5]
print("Metadata file path: ", metadata_file)

# Check if video file exists
if os.path.isfile(video_file):
    print("Video file exists.")
else:
    error = "Designated video file does not exist! exiting..."
    sys.exit(error)
    
# Check if metadata file exists
if os.path.isfile(metadata_file):
    metadata = pd.read_csv(metadata_file)
    print("Metadata file exists.")
else:
    error = "Designated metadata file does not exist! exiting..."
    sys.exit(error)
    
# Import values from metadata
start_index_hr = float(metadata.loc[metadata['Variable'] == 'Video Start Index']['Value'].tolist()[0])
end_index_hr = float(metadata.loc[metadata['Variable'] == 'Video End Index']['Value'].tolist()[0])
downsample_factor = int(metadata.loc[metadata['Variable'] == 'Video Background Downsample Factor']['Value'].tolist()[0])

mean_avg_pixels = int(metadata.loc[metadata['Variable'] == 'Mean Avg Pixels']['Value'].tolist()[0])
min_delta_pixels = int(metadata.loc[metadata['Variable'] == 'Min Delta Pixels']['Value'].tolist()[0])
max_delta_pixels = int(metadata.loc[metadata['Variable'] == 'Max Delta Pixels']['Value'].tolist()[0])
    
frame_pts = pd.read_csv(f'{output_dir}/Motion_Tracking/Motion_Intermediates/{batch_id}_frame_rate.txt', names = ["pts", "frame"])
pts = frame_pts['pts']*1000
pts = pts.astype(int)


# make the necessary folders:

os.makedirs(f'{output_dir}/Motion_Tracking/Motion_Outputs', exist_ok=True)
os.makedirs(f'{output_dir}/Motion_Tracking/Motion_Intermediates', exist_ok=True)


# Find the frame number corresponding to the start and end indices
# Don't include first five minutes
if start_index_hr < 0.0834:
    start_index_hr = 0.0834
start_timestamp = pts[pts >= start_index_hr * 60 * 60 * 1000].min()
start_index_frame = pts[pts == start_timestamp].index[0]

end_timestamp = pts[pts >= end_index_hr * 60 * 60 * 1000].min()
end_index_frame = pts[pts == end_timestamp].index[0]

print("Start index timestamp (ms): ", start_timestamp)
print("Start index frame: ", start_index_frame)
print("End index timestamp (ms): ", end_timestamp)
print("End index frame: ", end_index_frame)

# Get pixel values & background frames
avg_pixels, pixel_frame_ids, background_frames = fz.get_pixel_value_and_extract_frames(video_file, start_index_frame, end_index_frame, downsample_factor)
avg_pixels_data = pd.DataFrame({'frame': pixel_frame_ids,
                                'avg_pixel_value': avg_pixels})

# plot average pixels (to find night vs daytime)
plt.plot(avg_pixels)
plt.savefig(f'{output_dir}/Motion_Tracking/Motion_Intermediates/{batch_id}_avg_pixels_plot')

# find & plot change in average pixels (to find night vs daytime)
delta_pixels = []
for i in range(0, (len(avg_pixels)-1)):
    delta_pixels.append(avg_pixels[i] - avg_pixels[i+1])
plt.plot(delta_pixels)
plt.savefig(f'{output_dir}/Motion_Tracking/Motion_Intermediates/{batch_id}_delta_pixels_plot')

# Split into day / night if applicable
print(max(delta_pixels))
print(np.mean(avg_pixels))

proceed = True
if max(delta_pixels) > max_delta_pixels and min(delta_pixels) < min_delta_pixels: # If day1, night, & day2
    print("Day1, Night, Day2")
    day1_start = start_index_frame
    day1_end = start_index_frame + np.where(np.array(delta_pixels) > max(delta_pixels)/2)[0][0] 
    night_start = day1_end + 1
    night_end = start_index_frame + np.where(np.array(delta_pixels) < min(delta_pixels)/2)[0][0]
    day2_start = night_end + 1
    day2_end = end_index_frame
elif max(delta_pixels) > max_delta_pixels: # If day1 & night
    print("Day1, Night")
    day1_start = start_index_frame
    day1_end = start_index_frame + np.where(np.array(delta_pixels) > max(delta_pixels)/2)[0][0] 
    night_start = day1_end + 1
    night_end = end_index_frame
    day2_start = 'NA'
    day2_end = 'NA'
elif min(delta_pixels) < min_delta_pixels: # If night & day2
    print("Night, Day2")
    day1_start = 'NA'
    day1_end = 'NA'
    night_start = start_index_frame
    night_end = start_index_frame + np.where(np.array(delta_pixels) < min(delta_pixels)/2)[0][0]
    day2_start = night_end + 1
    day2_end = end_index_frame
elif np.mean(avg_pixels) > mean_avg_pixels: # If just day1
    print("Only Day1")
    day1_start = start_index_frame
    day1_end = end_index_frame
    night_start = 'NA'
    night_end = 'NA'
    day2_start = 'NA'
    day2_end = 'NA'
elif np.mean(avg_pixels) < mean_avg_pixels: # If just night
    day1_start = 'NA'
    day1_end = 'NA'
    night_start = start_index_frame
    night_end = end_index_frame
    day2_start = 'NA'
    day2_end = 'NA'
    proceed = False

# start & stop times (save for later)
# start & stop times (save for later)
segment_data = pd.DataFrame({'names': ['day1_start', 'day1_end',
                                       'night_start', 'night_end',
                                       'day2_start', 'day2_end'],
                             'frame': [day1_start, day1_end,
                                       night_start, night_end,
                                       day2_start, day2_end]})

if proceed == True:
    # which background frames to use
    if day1_start != 'NA':
        background_frames_day1_start = 10
        print(background_frames_day1_start)
        background_frames_day1_end = int((day1_end-day1_start)/downsample_factor) - 10
        print(background_frames_day1_end)
    if day2_start != 'NA' and day1_start != 'NA':
        background_frames_day2_start = int((day2_start-day1_start)/downsample_factor) + 10
        background_frames_day2_end = int((day2_end-day1_start)/downsample_factor) - 10
        print(background_frames_day2_start)
        print(background_frames_day2_end)
    elif day2_start != 'NA':
        background_frames_day2_start = int((day2_start-night_start)/downsample_factor) + 10
        background_frames_day2_end = int((day2_end-night_start)/downsample_factor) - 10
        print(background_frames_day2_start)
        print(background_frames_day2_end)
        

    # greyscale background frames
    background_frames_grey = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in background_frames]

    # remove very dark pixels from background frames
    frame_shape = background_frames_grey[0].shape[0]
    background_frames_filtered_day = []
    for i in tqdm(range(0, len(background_frames_grey))):
        background_frame_day = []
        for j in range(0, frame_shape):
            # Daytime filter (pixel value threshold = 120)
            s_day = pd.Series(background_frames_grey[i][j])
            s_day[s_day<120] = np.nan
            s_day = s_day.fillna(method='pad')
            background_frame_day.append(s_day.tolist())
        background_frames_filtered_day.append(np.array(background_frame_day))

    if day1_start != 'NA':
        print('day 1 number of background frames:', len(background_frames[background_frames_day1_start:background_frames_day1_end]))
        print(background_frames_day1_start)
        print(background_frames_day1_end)
        # Percentile background calculated over frames from day 1
        # 90%
        day1_background_img = np.nanpercentile(np.array(background_frames_filtered_day[background_frames_day1_start:background_frames_day1_end]), 90, axis=0)
        day1_background_img[np.isnan(day1_background_img)]=200
        day1_background_img = day1_background_img.astype('uint8')
        cv2.imwrite(f'{output_dir}/Motion_Tracking/Motion_Intermediates/{batch_id}_day1_background_img.png', day1_background_img)
        np.save(f'{output_dir}/Motion_Tracking/Motion_Intermediates/{batch_id}_day1_background_img.npy', day1_background_img)

    if day2_start != 'NA':
        print('day 2 number of background frames:', len(background_frames[background_frames_day2_start:background_frames_day2_end]))
        # Percentile background calculated over frames from day 2
        # 90%
        day2_background_img = np.nanpercentile(np.array(background_frames_filtered_day[background_frames_day2_start:background_frames_day2_end]), 90, axis=0)
        day2_background_img[np.isnan(day2_background_img)]=200
        day2_background_img = day2_background_img.astype('uint8')
        cv2.imwrite(f'{output_dir}/Motion_Tracking/Motion_Intermediates/{batch_id}_day2_background_img.png', day2_background_img)
        np.save(f'{output_dir}/Motion_Tracking/Motion_Intermediates/{batch_id}_day2_background_img.npy', day2_background_img)
        
# Output segmentation file to denote completion
segment_data.to_csv(f'{output_dir}/Motion_Tracking/Motion_Intermediates/{batch_id}_segment_data.csv', index=False)