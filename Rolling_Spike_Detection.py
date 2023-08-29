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

print(sys.argv)

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

# detect data points that are above a given threshold
if detect_spikes == 'True':
    print("Spike Detection has been set to occur")
    os.makedirs(f'{output_dir}/Event_Detection_Data/Spikes', exist_ok=True)
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
            threshold = average_value + 6.5*std
            if np.max(data) > threshold:
                amplitude = np.max(data)
                #print(f'threshold: {threshold} amplitude: {amplitude}')
                spike_times.append(start)
                spike_amplitudes.append(amplitude)

        Spike_Data = {'Spike Times': spike_times, 'Spike Amplitudes': spike_amplitudes}
        dataframe = pd.DataFrame(Spike_Data)
        os.makedirs(f'{output_dir}/Event_Detection_Data/Spikes/Rolling_Spikes/{current_channel}', exist_ok=True)
        dataframe.to_csv(f'{output_dir}/Event_Detection_Data/Spikes/Rolling_Spikes/{current_channel}/{current_channel}_rolling_spikes.csv', index=False)

# count seizure spikes if seizure detection is also set to true
        if detect_seizures == 'True':
            seizures_df = pd.read_csv(f'{output_dir}/Event_Detection_Data/Seizures/{current_channel}_Seizures.csv')
            spikes_df = pd.read_csv(f'{output_dir}/Event_Detection_Data/Spikes/Rolling_Spikes/{current_channel}/{current_channel}_rolling_spikes.csv')
            
            spikes_df['Seizure?'] = ''
            for index, seizures_row in seizures_df.iterrows():
                start_time = seizures_row['Start Times']
                end_time = seizures_row['End Times']
                mask = (spikes_df['Spike Times'] >= start_time) & (spikes_df['Spike Times'] <= end_time)
                spikes_df.loc[mask, 'Seizure?'] = 'Seizure'
            spikes_df['Seizure?'].fillna('', inplace=True)
            spikes_df.to_csv(f'{output_dir}/Event_Detection_Data/Spikes/Rolling_Spikes/{current_channel}/{current_channel}_rolling_spikes_with_seizures.csv', index=False)
