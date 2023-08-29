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
plt.rcParams["figure.figsize"] = (15,5)

input_dir = sys.argv[1]
print(input_dir)
output_dir = sys.argv[2]
print(output_dir)
batch_id = sys.argv[3]
print(batch_id)
current_channel = sys.argv[4]
print(current_channel)

# check if metadata file exists
metadata_file = input_dir + "/" + batch_id + "_metadata.csv"
if os.path.isfile(metadata_file):
    metadata = pd.read_csv(metadata_file)
    print("Metadata file exists.")
else:
    error = "Designated metadata file does not exist! exiting..."
    sys.exit(error)

# import values from metadata
mouse_names = ast.literal_eval(metadata.loc[metadata['Variable'] == 'Mouse IDs']['Value'].tolist()[0])
mouse_crops = ast.literal_eval(metadata.loc[metadata['Variable'] == 'Mouse Coordinates']['Value'].tolist()[0])

# format list of crops so it can be called in the function
mouse_crops_list = []
for i in mouse_crops:
    if len(i) != 4:
        print(f"Warning: Invalid entry in mouse_crops: {i}")
        continue
    
    mouse_crop_i = ','.join(str(coord) for coord in i)
    mouse_crop_i = list(map(int, mouse_crop_i.split(',')))
    mouse_crops_list.append(mouse_crop_i)

print(mouse_crops_list)

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
end_frame = max(day1_start, day1_end, night_start, night_end, day2_start, day2_end)
num_frames = int(end_frame) + 1

video_file = f"{input_dir}/Raw_EEG_Videos/{batch_id}.wmv"

def subtract_background(video_filename, name, fps, num_frames, mice_names, mouse_crop, start, end, day, background_img, output_dir):
    
    # input video
    vid_in = cv2.VideoCapture(video_filename)
    
    # lists for output videos
    vid_out = []
    
    for m in range(0,len(mice_names)):
        mouse = mice_names[m]
        print(mouse)
        x1, y1, x2, y2 = mouse_crop[m]

        # Calculate the width and height of the cropped region
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        print(width)
        print(height)
        # output video
        vid_out.append(cv2.VideoWriter(f'{output_dir}/Motion_Tracking/Motion_Intermediates/{name}_{mouse}_{day}_minus_background.mp4',cv2.VideoWriter_fourcc(*'FMP4'), fps, (width, height)))
        
    for i in tqdm(range(1,num_frames+1)):
        success,image = vid_in.read()
        if success == True:
            if (start <= i <= end):
                image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                diff = 255-(255.*np.maximum(background_img/255.-image_grey/255.,0)).astype('uint8')
                diff_contrast = cv2.convertScaleAbs(diff, alpha=1.3, beta=-70)
                diff_contrast_blur = cv2.GaussianBlur(diff_contrast,(3,3),0)
                diff_contrast_blur_color = cv2.cvtColor(diff_contrast_blur, cv2.COLOR_GRAY2BGR)
                
                for m in range(0, len(mice_names)):
                    x1, y1, x2, y2 = mouse_crop[m]
                    vid_out[m].write(diff_contrast_blur_color[y1:y2, x1:x2])
                    #vid_out[m].write(diff_contrast_blur_color[100:300, 200:400])

    for m in range(0,len(mice_names)):   
        vid_out[m].release()

# Import background images
if not math.isnan(day1_start):
    day1_name = output_dir + "/Motion_Tracking/Motion_Intermediates/" + batch_id + "_day1_background_img.npy"
    day1_background_img = np.load(day1_name)
    # subtract background from each segment
    print("Subtracting background for day 1..")
    subtract_background(video_file, batch_id, 60, num_frames, mouse_names, mouse_crops_list, day1_start, day1_end, "day1", day1_background_img, output_dir)

if not math.isnan(day2_start):
    day2_name = output_dir + "/Motion_Tracking/Motion_Intermediates/" + batch_id + "_day2_background_img.npy"
    day2_background_img = np.load(day2_name)
    # subtract background from each segment
    print("Subtracting background for day 2..")
    subtract_background(video_file, batch_id, 60, num_frames, mouse_names, mouse_crops_list, day2_start, day2_end, "day2", day2_background_img, output_dir)

# Write output file to denote completion
messages = [f'Background subtraction complete for batch {batch_id}']
message_pd = pd.DataFrame({'message': messages})
message_pd.to_csv(f'{output_dir}/Motion_Tracking/Motion_Intermediates/{batch_id}_background_subtraction_complete.csv', index=False)
