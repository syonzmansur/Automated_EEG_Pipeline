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
import csv

from scipy import signal
from scipy.fftpack import fft
from scipy.signal import welch
from yasa import sleep_statistics
from yasa import transition_matrix
from scipy.signal import welch

# Check if metadata file exists
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

metadata_file = input_dir + "/" + batch_id + "_metadata.csv"
if os.path.isfile(metadata_file):
    metadata = pd.read_csv(metadata_file)
    print("Metadata file exists.")
else:
    error = "Designated metadata file does not exist! exiting..."
    sys.exit(error)

# call variables from metadata
unused_channels = ast.literal_eval(metadata.loc[metadata['Variable'] == 'Unused Channels']['Value'].tolist()[0])
empty_channels = ast.literal_eval(metadata.loc[metadata['Variable'] == 'Empty Channels']['Value'].tolist()[0])
all_but_current_channel = ast.literal_eval(metadata.loc[metadata['Variable'] == 'Channel Names']['Value'].tolist()[0])

detect_spikes = metadata.loc[metadata['Variable'] == 'spikes']['Value'].tolist()[0]
detect_swds = metadata.loc[metadata['Variable'] == 'swds']['Value'].tolist()[0]
detect_seizures = metadata.loc[metadata['Variable'] == 'seizures']['Value'].tolist()[0]
detect_spindles = metadata.loc[metadata['Variable'] == 'spindles']['Value'].tolist()[0]
detect_threshold_spikes = metadata.loc[metadata['Variable'] == 'Threshold Spikes']['Value'].tolist()[0]
detect_rolling_spikes = metadata.loc[metadata['Variable'] == 'Rolling Spikes']['Value'].tolist()[0]

video_start_index = float(metadata.loc[metadata['Variable'] == 'Video Start Index']['Value'].tolist()[0])
video_end_index = float(metadata.loc[metadata['Variable'] == 'Video End Index']['Value'].tolist()[0])

# more metadata files-- these will be used to get the brainstates for the plots
sample_rate = float(metadata.loc[metadata['Variable'] == 'Sample Frequency']['Value'].tolist()[0])
epoch_length = int(metadata.loc[metadata['Variable'] == 'Epoch Time']['Value'].tolist()[0])

# calculate the number of samples per epoch
samples_per_epoch = int(sample_rate * epoch_length)

# make folder to save everything

os.makedirs(f'{output_dir}/Fine_Mapped_Spectral_Analysis/{current_channel}', exist_ok=True)

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
print(f' the value to find is: {value_to_find}')
index_found = find_and_remove_value(all_but_current_channel, value_to_find)

if index_found is not None:
    print(f"The value {value_to_find} was found at index {index_found} and has been removed.")
else:
    print(f"The value {value_to_find} was not found in the list.")
    
print("Updated list:", all_but_current_channel)

raw.drop_channels(all_but_current_channel)

print(f'The channel being analyzed is: {raw.ch_names}')

raw.crop(tmin = 3600 * video_start_index, tmax = 3600 * video_end_index - 1)
# Apply a bandpass filter from 0.5 to 40 Hz
raw.filter(1, 30)

# set eventless equal to all of the data
eventless = raw.get_data(0, 0, raw.n_times)
print(len(eventless))

# call combined hypnogram data -- has all of the brainstate info
hypno_data = f"{output_dir}/Hypnogram_Data/{current_channel}/{current_channel}_combined_hypnogram.csv"

# new arrays for epoch number and brainstate
epoch_number_array = []
brain_state_array = []
with open(hypno_data, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    
    # get the epoch number and corresponding brain state; append to the arrays
    for row in csv_reader:
        epoch_number_array.append(row[0])
        brain_state_array.append(row[-1])

# multiply each value by sample rate * epoch = samples_per_ epoch because that's the amount of samples in each epoch (the brain states are currently sorted by epoch)

extended_array_epoch = []
extended_array_brain = []
for value in epoch_number_array:
    extended_array_epoch.extend([value] * samples_per_epoch)
    
for value in brain_state_array:
    extended_array_brain.extend([value] * samples_per_epoch)


    
# get motion score data

motion_score_file = f"{output_dir}/Motion_Tracking/Motion_Outputs/{current_channel_mouse_id}/{current_channel_mouse_id}_motion.csv"
motion_scores = []
with open(motion_score_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    
    # get the epoch number and corresponding brain state; append to the arrays
    for row in csv_reader:
        motion_scores.append(row[2])

# set columns for checking row length
start_column = -1
end_column = 0

# remove seizures

# check if users opted to screen for seizures
if detect_seizures == 'True':
    # call seizure data
    seizure_file = f"{output_dir}/Event_Detection_Data/Seizures/{current_channel}_Seizures.csv"

    # create arrays for start and end values
    start_values = []
    end_values = []
    
    target_column_index = -1
    with open(seizure_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        # iterate through each row in the CSV file
        for row in csv_reader:
            # check if the row has enough columns
            if target_column_index < len(row):
                start_values.append(row[start_column+1])

    target_column_index = 0
    with open(seizure_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        # iterate through each row in the CSV file
        for row in csv_reader:
            # check if the row has enough columns
            if target_column_index < len(row):
                end_values.append(row[end_column+1])

    # set start and end values, then delete the values in between those indexes from each array
    for i in range(len(start_values)):
        start = int(start_values[i])
        end = int(end_values[i])     
        numbers_between = np.arange(start, end + 1)
        for i in numbers_between:
            eventless[0,i] = 0
            extended_array_epoch[i] = 0
            extended_array_brain[i] = 0
            motion_scores[i] = 0


# spikes
if detect_spikes == 'True':
    if detect_rolling_spikes == 'True': 
        spike_file = f"{output_dir}/Event_Detection_Data/Spikes/Rolling_Spikes/{current_channel}/{current_channel}_rolling_spikes.csv"
        start_column = -1
        end_column = 0

        start_values = []
        with open(spike_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            # Iterate through each row in the CSV file
            for row in csv_reader:
                # Check if the row has enough columns
                if target_column_index < len(row):
                    start_values.append(row[start_column+1])

        end_values = []
        for spike in start_values:
            time = int(spike) + 1
            end_values.append(time)

# set start and end values, then delete the values in between those indexes from each array
        for i in range(len(start_values)):
            start = int(start_values[i])
            end = int(end_values[i])
            numbers_between = np.arange(start, end + 1)
            for i in numbers_between:
                eventless[0,i] = 0
                extended_array_epoch[i] = 0
                extended_array_brain[i] = 0
                motion_scores[i] = 0
                
    else:
        spike_file = f"{output_dir}/Event_Detection_Data/Spikes/Threshold_Spikes/{current_channel}/{current_channel}_threshold_spikes.csv"
        start_column = -1
        end_column = 0

        start_values = []
        with open(spike_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            # Iterate through each row in the CSV file
            for row in csv_reader:
                # Check if the row has enough columns
                if target_column_index < len(row):
                    start_values.append(row[start_column+1])

        end_values = []
        for spike in start_values:
            time = int(spike) + 1
            end_values.append(time)

# set start and end values, then delete the values in between those indexes from each array
    for i in range(len(start_values)):
        start = int(start_values[i])
        end = int(end_values[i])
        numbers_between = np.arange(start, end + 1)
        for i in numbers_between:
            eventless[0,i] = 0
            extended_array_epoch[i] = 0
            extended_array_brain[i] = 0
            motion_scores[i] = 0


# swds

if detect_swds == 'True':

    swd_file = f"{output_dir}/Event_Detection_Data/SWDs/{current_channel}_SWD_events.csv"

    with open(swd_file, 'r') as csvfile:
        # Create a CSV reader object
        csvreader = csv.reader(csvfile)

        next(csvreader)

        # Loop through the rows
        for row_index, row in enumerate(csvreader, start=1):
            # Check if it's the second row
            if row_index == 1:
                # Get the value in the second column (index 1)
                value = row[1]
                print(value)
            
    if value == 'No swds found':
        print("No Spindles")
        
    else:
        target_column_index = 0
        start_values = []
        with open(swd_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            # Iterate through each row in the CSV file
            for row in csv_reader:
                # Check if the row has enough columns
                if target_column_index < len(row):
                    start_values.append(float(row[start_column+1]))

    # convert values to ms (YASA does not do this so we have to manually do it) - yasa was used to get spindle and swd data
        start_values_ms = []
        for value in start_values:
            start_values_ms.append(int(value * 1000))
        print(start_values_ms)

        target_column_index = 2
        end_values = []
        with open(swd_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            # Iterate through each row in the CSV file
            for row in csv_reader:
                # Check if the row has enough columns
                if target_column_index < len(row):
                    end_values.append(float(row[end_column+1]))

    # convert values to ms (YASA does not do this so we have to manually do it) - yasa was used to get spindle and swd data
        end_values_ms = []
        for value in end_values:
            end_values_ms.append(int(value * 1000))
        print(end_values_ms)

        for i in range(len(start_values)):
            start = int(start_values[i])
            end = int(end_values[i])
            numbers_between = np.arange(start, end + 1)
            for i in numbers_between:
                eventless[0,i] = 0
                extended_array_epoch[i] = 0
                extended_array_brain[i] = 0
                motion_scores[i] = 0

# spindles

if detect_spindles == 'True':

    spindle_file = f"{output_dir}/Event_Detection_Data/Spindles/{current_channel}_Spindle_events.csv"
    
    with open(spindle_file, 'r') as csvfile:
    # Create a CSV reader object
        csvreader = csv.reader(csvfile)

        next(csvreader)

        # Loop through the rows
        for row_index, row in enumerate(csvreader, start=1):
            # Check if it's the second row
            if row_index == 1:
                # Get the value in the second column (index 1)
                value = row[1]
                print(value)
            
    if value == 'No spindles found':
        print("No Spindles")
        
    else:

        target_column_index = 0
        start_values = []
        with open(spindle_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            # Iterate through each row in the CSV file
            for row in csv_reader:
                # Check if the row has enough columns
                if target_column_index < len(row):
                    start_values.append(float(row[start_column+1]))
    # convert values to ms (YASA does not do this so we have to manually do it) - yasa was used to get spindle and swd data
        start_values_ms = []
        for value in start_values:
            start_values_ms.append(int(value * 1000))
        print(start_values_ms)

        target_column_index = 2
        end_values = []
        with open(spindle_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            # Iterate through each row in the CSV file
            for row in csv_reader:
                # Check if the row has enough columns
                if target_column_index < len(row):
                    end_values.append(float(row[end_column+1]))
    # convert values to ms (YASA does not do this so we have to manually do it) - yasa was used to get spindle and swd data
        end_values_ms = []
        for value in end_values:
            end_values_ms.append(int(value * 1000))
        print(end_values_ms)

        for i in range(len(start_values)):
            start = int(start_values[i])
            end = int(end_values[i])
            numbers_between = np.arange(start, end + 1)
            for i in numbers_between:
                eventless[0,i] = 0
                extended_array_epoch[i] = 0
                extended_array_brain[i] = 0
                motion_scores[i] = 0

# turn the arrays into a df
print(len(eventless[0]))
print(len(extended_array_brain))

if len(eventless[0]) < len(extended_array_brain):
    nan_padding = np.full(len(extended_array_brain) - len(eventless[0]), np.nan)
    eventless = np.concatenate((eventless[0], nan_padding))
elif len(extended_array_brain) < len(eventless[0]):
    nan_padding = np.full(len(eventless[0]) - len(extended_array_brain), np.nan)
    extended_array_brain = np.concatenate((extended_array_brain, nan_padding))
    
motion_scores_0 = motion_scores != 0
motion_scores = motion_scores[motion_scores_0]


extended_array_epoch_0 = extended_array_epoch != 0
extended_array_epoch = extended_array_epoch[extended_array_epoch_0]

extended_array_brain_0 = extended_array_brain != 0
extended_array_brain = extended_array_brain[extended_array_brain_0]

eventless_0 = eventless != 0
eventless = eventless[eventless_0]

print(eventless[0])
print(motion_scores)
        
data = {'EEG Data': eventless, 'Brain State': extended_array_brain}
df = pd.DataFrame(data)

print(df)

# new arrays to hold and eventually graph the data
wake_array = []
sleep_array = []
blank_array = []

# loop through df and make arrays that correspond to each brain state
for index, row in df.iterrows():
    if row['Brain State'] == 'Wake':
        wake_array.append(row['EEG Data'])
    elif row['Brain State'] == 'Sleep':
        sleep_array.append(row['EEG Data'])
    elif row['Brain State'] == '':
        blank_array.append(row['EEG Data'])

# Convert the lists to arrays (if needed)
wake_array = np.array(wake_array)
sleep_array = np.array(sleep_array)
blank_array = np.array(blank_array)

# plot and save
fig, ax = plt.subplots(figsize=[15, 5])
ax.plot(wake_array)
plt.title(f"EEG Signal for Wake Brain State")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (V)")
plt.savefig(f'{output_dir}/Fine_Mapped_Spectral_Analysis/{current_channel}/{current_channel}_Wake_Fine_Mapped_Spectral_Analysis.png')
plt.show()

# plot and save
plt.figure(figsize=(12, 4))
plt.plot(sleep_array)
plt.title(f"EEG Signal for Sleep Brain State")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (V)")
plt.savefig(f'{output_dir}/Fine_Mapped_Spectral_Analysis/{current_channel}/{current_channel}_Sleep_Fine_Mapped_Spectral_Analysis.png')
plt.show

# plot and save
plt.figure(figsize=(12, 4))
plt.plot(eventless)
plt.title(f"EEG Signal for All Brain States")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (V)")
plt.savefig(f'{output_dir}/Fine_Mapped_Spectral_Analysis/{current_channel}/{current_channel}_full_Fine_Mapped_Spectral_Analysis.png')
plt.show


#####################################################################################################################
# recalculate spectral outputs from hypnogram data but with this new data!
#####################################################################################################################
os.makedirs(f'{output_dir}/Fine_Mapped_Spectral_Analysis/{current_channel}/Fine_Hypnogram_Data', exist_ok=True)

video_start_index = float(metadata.loc[metadata['Variable'] == 'Video Start Index']['Value'].tolist()[0])
video_end_index = float(metadata.loc[metadata['Variable'] == 'Video End Index']['Value'].tolist()[0])

Minimum_Wake_State = float(metadata.loc[metadata['Variable'] == 'Minimum Wake State']['Value'].tolist()[0])
Minimum_Delta_Theta_Ratio = float(metadata.loc[metadata['Variable'] == 'Minimum Delta Theta Ratio']['Value'].tolist()[0])
Brain_State_Z_Score_Threshold = float(metadata.loc[metadata['Variable'] == 'Brain State Z-Score threshold']['Value'].tolist()[0])


# calculate the number of samples in 1 hour
samples_per_hour = sample_rate * 3600

# samples per epoch was calculated above

# calculate the start and end indices of the data to use
start_index = int(samples_per_hour * video_start_index)
end_index = int(samples_per_hour * video_end_index)
                   
# create an empty list to store the Motion scores for the selected time range
motion_scores_array = []

df = pd.DataFrame({'motion': motion_scores}, index=[0])

# Iterate through motion score data by epoch
for i in range(start_index, end_index, samples_per_epoch):
    #Get data for current epoch
    epoch_data = df.iloc[i:i+samples_per_epoch]
    # Calculate the average motion score for the epoch
        #convert all the '' values to Nan so you can take the mean
    df['motion'] = pd.to_numeric(df['motion'], errors='coerce')
    motion_score = df['motion'].mean()
    # add the score to the list
    motion_scores_array.append(motion_score)

# Calculate the number of samples per epoch
samples_per_epoch = int(sample_rate * epoch_length)

# Initialize lists to store the power values
delta_power = []
theta_power = []
delta_theta_ratio = []
wideband_power = []
z_scores = []

# Iterate through EEG data by epoch
for i in range(0, len(raw.get_data()[0]), samples_per_epoch):
    # Get the data for the current epoch
    epoch_data = raw.get_data()[0][i:i+samples_per_epoch]

    # Perform FFT on the epoch data
    fft_data = np.abs(fft(epoch_data))

    # Calculate the relative power in the delta band (0.5-4 Hz)
    delta_band = np.where((fft_data >= 0.5) & (fft_data <= 4))
    delta_power.append(np.sum(fft_data[delta_band])/np.sum(fft_data))

    # Calculate the relative power in the theta band (5-8 Hz)
    theta_band = np.where((fft_data >= 5) & (fft_data <= 8))
    theta_power.append(np.sum(fft_data[theta_band])/np.sum(fft_data))
    
    # Calculate the ratio of delta power to theta power
    delta_theta_ratio.append(delta_power[-1]/theta_power[-1])

    # Calculate the wideband power (0.5-200 Hz)
    wideband_band = np.where((fft_data >= 0.5) & (fft_data <= 200))
    wideband_power.append(np.sum(fft_data[wideband_band])/np.sum(fft_data))

    # Calculate the z-score of the wideband power
    z_scores.append((wideband_power[-1]-np.mean(wideband_power))/np.std(wideband_power))

# Determine the maximum
max_length = max(len(delta_power), len(theta_power), len(delta_theta_ratio), len(wideband_power), len(z_scores), len(motion_scores_array))

# add Nan to make them the same length
delta_power += [float('nan')] * (max_length - len(delta_power))
theta_power += [float('nan')] * (max_length - len(theta_power))
delta_theta_ratio += [float('nan')] * (max_length - len(delta_theta_ratio))
wideband_power += [float('nan')] * (max_length - len(wideband_power))
z_scores += [float('nan')] * (max_length - len(z_scores))
motion_scores_array += [float('nan')] * (max_length - len(motion_scores_array))


# Add the power values to the BrainStateScore DataFrame
BrainStateScore = pd.DataFrame({"Motion": motion_scores_array})
BrainStateScore["Delta_Power"] = delta_power
BrainStateScore["Theta_Power"] = theta_power
BrainStateScore["Delta_Theta_Ratio"] = delta_theta_ratio
BrainStateScore["Wideband_Power"] = wideband_power
BrainStateScore["Z_Scores"] = z_scores 

BrainStateScore["BrainState"] = ""

for i in range(len(BrainStateScore)):
    if BrainStateScore.loc[i, "Motion"] >= Minimum_Wake_State:
        BrainStateScore.loc[i, "BrainState"] = "Wake"
    elif BrainStateScore.loc[i, "Motion"] < Minimum_Wake_State and BrainStateScore.loc[i, "Delta_Theta_Ratio"] >= Minimum_Delta_Theta_Ratio:
        BrainStateScore.loc[i, "BrainState"] = "REM"
    elif BrainStateScore.loc[i, "Motion"] < Minimum_Wake_State and BrainStateScore.loc[i, "Delta_Theta_Ratio"] < Minimum_Delta_Theta_Ratio and BrainStateScore.loc[i, "Z_Scores"] > 1:
        BrainStateScore.loc[i, "BrainState"] = "Resting Wakefulness"
    elif BrainStateScore.loc[i, "Motion"] < Minimum_Wake_State and BrainStateScore.loc[i, "Delta_Theta_Ratio"] < Minimum_Delta_Theta_Ratio and BrainStateScore.loc[i, "Z_Scores"] < Brain_State_Z_Score_Threshold:
        BrainStateScore.loc[i, "BrainState"] = "NREM"


#%% Generate a hypnogram text file which can be used in conjuction with event detector script and spectral analysis script

####Hypnogram 1 - Active wake, resting wake, NREM, REM
Hypnogram_1 = BrainStateScore[["BrainState"]]

# Save the DataFrame as a CSV file

####Hypnogram 2 - Wake versus Sleep
# Convert values in the BrainState column to the desired values
BrainStateScore["BrainState"] = BrainStateScore["BrainState"].replace(["Wake", "Resting Wakefulness"], "Wake")
BrainStateScore["BrainState"] = BrainStateScore["BrainState"].replace(["NREM", "REM"], "Sleep")

# Create a new DataFrame with only the BrainState column
Hypnogram_2 = BrainStateScore[["BrainState"]]

#combine dataframes

#%% Optional: Create sample plots from each brain state

# Extract the EEG data
eeg_data = raw.get_data()[0]

# Calculate the number of samples per epoch
samples_per_epoch = int(sample_rate * epoch_length)

# Get the unique brain states
unique_states = BrainStateScore["BrainState"].unique()

for state in unique_states:
    
    # Get the indices of the rows that correspond to the current brain state
    state_indices = BrainStateScore[BrainStateScore["BrainState"] == state].index

    # Choose a random index from the list of indices
    random_index = np.random.choice(state_indices)

    # Get the EEG data for the current epoch
    epoch_data = eeg_data[random_index * samples_per_epoch : (random_index + 1) * samples_per_epoch]

    # Plot the EEG signal
    plt.figure(figsize=(12, 4))
    plt.plot(epoch_data)
    plt.title(f"EEG Signal for {state} Brain State")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (V)")
    plt.savefig(f'{output_dir}/Fine_Mapped_Spectral_Analysis/{current_channel}/Fine_Hypnogram_Data/{current_channel}_{state}_EEG_signal.png')
    # Calculate Welch's periodogram
    f, Pxx_den = welch(epoch_data, sample_rate, nperseg=samples_per_epoch)

    # Plot Welch's periodogram
    plt.figure(figsize=(12, 4))
    plt.semilogy(f, Pxx_den)
    plt.title(f"Welch's Periodogram for {state} Brain State")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (V^2/Hz)")
    plt.savefig(f'{output_dir}/Fine_Mapped_Spectral_Analysis/{current_channel}/Fine_Hypnogram_Data/{current_channel}_{state}_welch_pdg.png')

    # Calculate spectrogram
    #f, t, Sxx = signal.spectrogram(epoch_data, sample_rate, nperseg=samples_per_epoch)

    # Plot Spectrogram
    #plt.figure(figsize=(12, 4))
    #plt.pcolormesh(t, f, Sxx)
    #plt.title(f"Spectrogram for {state} Brain State")
    #plt.xlabel("Time (s)")
    #plt.ylabel("Frequency (Hz)")
    #plt.yscale("log")
    #plt.xlim(0, 1200)
    #plt.colorbar().set_label("Amplitude")
    
    #Find and Plot Power Spectrum for each Brain state
    win = int(4 * sample_rate)  # Window size is set to 4 seconds
       
    
    def find_series_of_repeats(data_frame, column_index, target_phrase):
        # Initialize variables to store the indices of the current series
        start_index = None
        end_index = None
        series_indices = []

        # Iterate through the DataFrame rows
        for index, row in data_frame.iterrows():
            # Check if the target phrase is found in the 7th column (index 6)
            if str(row[column_index]).find(target_phrase) != -1:
                # If start_index is not set, it means this is the beginning of a new series
                if start_index is None:
                    start_index = index

                # Always update the end_index until the loop completes
                end_index = index
            else:
                # If the phrase is not found in the current row, it means the series ended
                if start_index is not None:
                    series_indices.append((start_index, end_index))
                    start_index = None
                    end_index = None

        # Check if a series ends at the last row of the DataFrame
        if start_index is not None:
            series_indices.append((start_index, end_index))

        return series_indices

    # Replace '7th_column_name' with the actual name of your 7th column
    seventh_column_index = 6

    # Replace 'target_phrase' with the phrase you want to find
    target_phrase = f'{state}'

    series_indices = find_series_of_repeats(BrainStateScore, seventh_column_index, target_phrase)

    start_values = []
    end_values = []
    if series_indices:
        for i, (start, end) in enumerate(series_indices, 1):
            print(f"Series {i}: '{target_phrase}' repeats from index {start} to index {end}.")
            start_values.append(start)
            end_values.append(end)     
    else:
        print(f"'{target_phrase}' was not found in the 7th column.")

    print(start_values)
    print(end_values)

    psd_data = []
    for i in range(len(start_values)):
        if start_values[i] == end_values[i]:
            data = 0
            psd_data.append(data)
        else:
            data = raw.get_data(0, start_values[i] * samples_per_epoch, end_values[i] * samples_per_epoch)
            psd_data.append(data)
    psd_data
    
    all_have_more_than_one_dimension = all(isinstance(array, np.ndarray) and array.ndim > 1 for array in psd_data)
    
    if all_have_more_than_one_dimension:
    
        psd_data_input = [np.hstack(psd_data)]
        desired_array = np.array(psd_data_input)

        desired_array_2 = np.squeeze(desired_array)

        desired_array_3 = [desired_array_2]
        desired_array_3

        freqs, psd = welch(desired_array_3, sample_rate, nperseg=win, average='median')

        print(freqs.shape, psd.shape)  # psd has shape (n_channels, n_frequencies)

        ##############OPTIONAL: Plot Power Spectrum for Each Channel##################
        #Plot 1 of the power spectrum 
        plt.plot(freqs, psd[0], 'k', lw=2)
        plt.fill_between(freqs, psd[0], cmap='Spectral') #This is a plot of channel 1
        plt.xlim(0, 50)
        plt.yscale('log')
        sns.despine()
        plt.title(f'{raw.ch_names[0]} Plot 1 PSD for {state}')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD log($uV^2$/Hz)');
        plt.savefig(f'{output_dir}/Fine_Mapped_Spectral_Analysis/{current_channel}/Fine_Hypnogram_Data/{current_channel}_{state}_PSD_1.png')

        # Plot 2 of the power spectrum
        sns.set(font_scale=1.2, style='white')
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, psd[0], color='k', lw=2) #This is a plot of channel 1
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density ($uV^2$/Hz)')
        plt.ylim([0, psd.max() * 1.1])
        plt.title(f"PSD for {state} state")
        plt.xlim([0, freqs.max()])
        #plt.xlim([0, 20])
        sns.despine()
        plt.savefig(f'{output_dir}/Fine_Mapped_Spectral_Analysis/{current_channel}/Fine_Hypnogram_Data/{current_channel}_{state}_PSD_2.png')
