#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yasa
import pandas as pd
import ast
import sys
import time

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
    
# define variables from setup script metadata file
baseline_window_step = int(metadata.loc[metadata['Variable'] == 'Baseline Window Step']['Value'].tolist()[0])
unused_channels = ast.literal_eval(metadata.loc[metadata['Variable'] == 'Unused Channels']['Value'].tolist()[0])
empty_channels = ast.literal_eval(metadata.loc[metadata['Variable'] == 'Empty Channels']['Value'].tolist()[0])
all_but_current_channel = ast.literal_eval(metadata.loc[metadata['Variable'] == 'Channel Names']['Value'].tolist()[0])

# event detection variables from metadata
detect_spikes = metadata.loc[metadata['Variable'] == 'spikes']['Value'].tolist()[0]
detect_swds = metadata.loc[metadata['Variable'] == 'swds']['Value'].tolist()[0]
detect_seizures = metadata.loc[metadata['Variable'] == 'seizures']['Value'].tolist()[0]
detect_spindles = metadata.loc[metadata['Variable'] == 'spindles']['Value'].tolist()[0]
detect_threshold_spikes = metadata.loc[metadata['Variable'] == 'Threshold Spikes']['Value'].tolist()[0]
detect_rolling_spikes = metadata.loc[metadata['Variable'] == 'Rolling Spikes']['Value'].tolist()[0]

video_start_index = float(metadata.loc[metadata['Variable'] == 'Video Start Index']['Value'].tolist()[0])
video_end_index = float(metadata.loc[metadata['Variable'] == 'Video End Index']['Value'].tolist()[0])

    
# Load EEG recording (12 animals)
raw = mne.io.read_raw_edf(f'{input_dir}/Raw_EEG_Files/{batch_id}.edf', preload=True)

raw.drop_channels(empty_channels)
raw.drop_channels(unused_channels)
print(f'your channels to be eventually analyzed are: {raw.ch_names}')

def find_and_remove_value(input_list, value):
    try:
        index = input_list.index(value)
        input_list.remove(value)
        return index
    except ValueError:
        return None

value_to_find = f'{current_channel}'
print(value_to_find)
index_found = find_and_remove_value(all_but_current_channel, value_to_find)

if index_found is not None:
    print(f"The value {value_to_find} was found at index {index_found} and has been removed.")
else:
    print(f"The value {value_to_find} was not found in the list.")
    
print("Updated list:", all_but_current_channel)

raw.drop_channels(all_but_current_channel)

print(time.time())

raw.crop(tmin = 3600 * video_start_index, tmax = 3600 * video_end_index - 1)


# Apply a bandpass filter from 0.5 to 40 Hz
raw.filter(0.5, 40)


# Define the baseline window size in samples (set to 20 sec)
baseline_window_size = int(round(raw.info['sfreq'] * 20))

os.makedirs(f'{output_dir}/Event_Detection_Data', exist_ok=True)
# spindle detection
if detect_spindles == 'True':
    os.makedirs(f'{output_dir}/Event_Detection_Data/Spindles', exist_ok=True)
    print("Spindle Detection has been set to occur")
    sf = raw.info['sfreq']
    data = raw.get_data(units = "mV")
    # Detect sleep spindles using parameters from Uyugen et al., 2019 Sleep (recommended)
    spindles = yasa.spindles_detect(data, sf, hypno=None, freq_sp=(10,15), freq_broad=(.5,50),
                                  duration=(0.5,10), min_distance=200, thresh={'rel_pow':0.2, 'corr':.65, 'rms':1.2})
    # append to df if spindles exist
    if spindles is not None:
        spindle_events = spindles.summary()
        spindle_events.to_csv(f'{output_dir}/Event_Detection_Data/Spindles/{current_channel}_Spindle_events.csv')
    # append No spindles found if no spindles are found
    else:
        data = {'Message': ['No spindles found']}
        df = pd.DataFrame(data)
        df.to_csv(f'{output_dir}/Event_Detection_Data/Spindles/{current_channel}_Spindle_events.csv')

