#!/usr/bin/env python
# coding: utf-8


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

# Apply a bandpass filter from 0.5 to 40 Hz
raw.filter(0.5, 40)


# Define the baseline window size in samples (set to 20 sec)
baseline_window_size = int(round(raw.info['sfreq'] * 20))

os.makedirs(f'{output_dir}/Event_Detection_Data', exist_ok=True)

if detect_spikes == 'True':
    print("Spike Detection has been set to occur")
    os.makedirs(f'{output_dir}/Event_Detection_Data/Spikes', exist_ok=True)
    if detect_threshold_spikes == 'True':
        # get thresholds from metadata
        spike_threshold = ast.literal_eval(metadata.loc[metadata['Variable'] == 'Spike Threshold']['Value'].tolist()[0])
        print(f'analysis has started for: {current_channel}')
        
        # create arrays for time and amplitudes
        spike_times = []
        spike_amplitudes = []
        
        # iterate through
        for start in range(0, raw.n_times, baseline_window_step):
            stop = start + baseline_window_step
            data = raw.get_data(start=start, stop=stop)
            print(start)
            if np.max(data) > spike_threshold[index_found]:
                amplitude = np.max(data)
                spike_times.append(start)
                spike_amplitudes.append(amplitude)
        Spike_Data = {'Spike Times': spike_times, 'Spike Amplitudes': spike_amplitudes}
        dataframe = pd.DataFrame(Spike_Data)
        os.makedirs(f'{output_dir}/Event_Detection_Data/Spikes/Threshold_Spikes/{current_channel}', exist_ok=True)
        dataframe.to_csv(f'{output_dir}/Event_Detection_Data/Spikes/Threshold_Spikes/{current_channel}/{current_channel}_threshold_spikes.csv', index=False)      
       
    if detect_rolling_spikes == 'True':
        spike_times = []
        spike_amplitudes = []
        for start in range(0, raw.n_times, baseline_window_step):
            stop = start + baseline_window_step
            data = raw.get_data(start=start, stop=stop)
            surrounding_avg = start + baseline_window_size
            avg_data_calc = raw.get_data(start=start, stop=surrounding_avg)
            average_value = np.mean(avg_data_calc)
            std = np.std(avg_data_calc)
            threshold = average_value + 4*std
            if np.max(data) > threshold:
                amplitude = np.max(data)
                #print(f'threshold: {threshold} amplitude: {amplitude}')
                spike_times.append(start)
                spike_amplitudes.append(amplitude)

        Spike_Data = {'Spike Times': spike_times, 'Spike Amplitudes': spike_amplitudes}
        dataframe = pd.DataFrame(Spike_Data)
        os.makedirs(f'{output_dir}/Event_Detection_Data/Spikes/Rolling_Spikes/{current_channel}', exist_ok=True)
        dataframe.to_csv(f'{output_dir}/Event_Detection_Data/Spikes/Rolling_Spikes/{current_channel}/{current_channel}_rolling_spikes.csv', index=False)      
        
if detect_swds == 'True':
    os.makedirs(f'{output_dir}/Event_Detection_Data/SWDs')
    print("SWD Detection has been set to occur")
    sf = raw.info['sfreq']
    data = raw.get_data(units = "V")
    # Detection of spike and wave discharges using parameters from Jeanne Paz in Das et al., 2023 (unpublished)
    swd = yasa.spindles_detect(data, sf, hypno=None, freq_sp=(6,10), freq_broad=(.5,50),
                              duration=(1,4), min_distance=200, thresh={'rel_pow':0.2, 'corr':.65, 'rms':2})
    swd_events = swd.summary()
    swd_events.to_csv(f'{output_dir}/Event_Detection_Data/SWDs/{current_channel}_SWD_events')

if detect_spindles == 'True':
    os.makedirs(f'{output_dir}/Event_Detection_Data/Spindles')
    print("Spindle Detection has been set to occur")
    sf = raw.info['sfreq']
    data = raw.get_data(units = "V")
    # Detect sleep spindles using parameters from Uyugen et al., 2019 Sleep (recommended)
    spindles = yasa.spindles_detect(data, sf, hypno=None, freq_sp=(10,15), freq_broad=(.5,50),
                                  duration=(0.5,10), min_distance=200, thresh={'rel_pow':0.2, 'corr':.65, 'rms':1.2})
    spindle_events = spindles.summary()
    spindle_events.to_csv(f'{output_dir}/Event_Detection_Data/Spindles/{current_channel}_Spindle_events')

if detect_seizures == 'True':
    os.makedirs(f'{output_dir}/Event_Detection_Data/Seizures, exist_ok=True)
    print("Seizure Detection has been set to occur")
    seizure_start_times = []
    seizure_end_times = []
    seizure_spike_counts = []
    for start in range(0, raw.n_times, baseline_window_step * 30):
        stop = start + baseline_window_step*10
        data = raw.get_data(0, start=start, stop=stop)
        surrounding_avg = start + baseline_window_size
        avg_data_calc = raw.get_data(start=start, stop=surrounding_avg)
        average_value = np.mean(avg_data_calc)
        std = np.std(avg_data_calc)
        threshold = average_value + 7 * std

        seizure_spikes = np.greater(data, threshold)
        spike_count = np.sum(seizure_spikes)
        if spike_count >= 10:
            first_spike = np.argmax(seizure_spikes)
            #seizure_spike_trues = np.where(seizure_spikes)[0]
            #print(seizure_spikes)
            #print(seizure_spike_trues)
            #last_spike = seizure_spike_trues[-1]
            #print(last_spike)
            first_spike_time = start + first_spike
            #print(first_spike_time)
            last_spike_time = first_spike_time + spike_count
            seizure_start_times.append(first_spike_time)
            seizure_end_times.append(last_spike_time)
            seizure_spike_counts.append(spike_count)
    Seizure_Data = {'Start Times': seizure_start_times, 'End Times': seizure_end_times, 'Spike Counts':seizure_spike_counts}
    Seizure_df = pd.DataFrame(Seizure_Data)
    dataframe.to_csv(f'{output_dir}/Event_Detection_Data/Seizures/{current_channel}_Seizures.csv', index=False)