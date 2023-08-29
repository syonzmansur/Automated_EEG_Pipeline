import pandas as pd
#need to input variables from metadata file


import numpy as np
import mne
import matplotlib.pyplot as plt
import yasa
import seaborn as sns
import sys
import os
import ast

from scipy import signal
from scipy.fftpack import fft
from scipy.signal import welch
from yasa import sleep_statistics
from yasa import transition_matrix
from scipy.signal import welch

sns.set(font_scale=1.2)

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

# current_channel_mouse_id = remove_after_period(current_channel)

metadata_file = input_dir + "/" + batch_id + "_metadata.csv"
if os.path.isfile(metadata_file):
    metadata = pd.read_csv(metadata_file)
    print("Metadata file exists.")
else:
    error = "Designated metadata file does not exist! exiting..."
    sys.exit(error)

df = pd.read_csv(f"{output_dir}/Motion_Tracking/Motion_Outputs/{current_channel_mouse_id}/{current_channel_mouse_id}_motion.csv")
    
# define variables from setup script metadata file

unused_channels = ast.literal_eval(metadata.loc[metadata['Variable'] == 'Unused Channels']['Value'].tolist()[0])
empty_channels = ast.literal_eval(metadata.loc[metadata['Variable'] == 'Empty Channels']['Value'].tolist()[0])
all_but_current_channel = ast.literal_eval(metadata.loc[metadata['Variable'] == 'Channel Names']['Value'].tolist()[0])

sample_rate = float(metadata.loc[metadata['Variable'] == 'Sample Frequency']['Value'].tolist()[0])
epoch_length = int(metadata.loc[metadata['Variable'] == 'Epoch Time']['Value'].tolist()[0])

video_start_index = float(metadata.loc[metadata['Variable'] == 'Video Start Index']['Value'].tolist()[0])
video_end_index = float(metadata.loc[metadata['Variable'] == 'Video End Index']['Value'].tolist()[0])

Minimum_Wake_State = float(metadata.loc[metadata['Variable'] == 'Minimum Wake State']['Value'].tolist()[0])
Minimum_Delta_Theta_Ratio = float(metadata.loc[metadata['Variable'] == 'Minimum Delta Theta Ratio']['Value'].tolist()[0])
Brain_State_Z_Score_Threshold = float(metadata.loc[metadata['Variable'] == 'Brain State Z-Score threshold']['Value'].tolist()[0])

#%% ####### Generate a file with a motion score for every 5 second epoch #######

########## Analyze 3 h of recording (from end of hour 1 (3600 s) to beginning of hour 4 (14400 s)) ###########

# calculate the number of samples in 1 hour
samples_per_hour = sample_rate * 3600

# calculate the number of samples in 5 seconds
samples_per_epoch = int(sample_rate * epoch_length)

# calculate the start and end indices of the data to use
start_index = int(samples_per_hour * video_start_index)
end_index = int(samples_per_hour * video_end_index)
                   
# create an empty list to store the Motion scores for the selected time range
motion_scores = []

# Iterate through motion score data by epoch
for i in range(start_index, end_index, samples_per_epoch):
    #Get data for current epoch
    epoch_data = df.iloc[i:i+samples_per_epoch]
    # Calculate the average motion score for the epoch
    motion_score = epoch_data["motion"].mean()
    # add the score to the list
    motion_scores.append(motion_score)


########## ALTERNATIVE: Run entire 24 h recording  ##############

# Create a new DataFrame to store the epoch scores
#BrainStateScore = pd.DataFrame(columns=["Motion"])

# Set the epoch length in seconds
#epoch_length = 5  # in seconds

# Calculate the number of samples per epoch
#samples_per_epoch = sample_rate * epoch_length

# Iterate through the data by epoch
#for i in range(0, len(df), samples_per_epoch):
    # Get the data for the current epoch
    #epoch_data = df.iloc[i:i+samples_per_epoch]
    # Calculate the average motion score for the epoch
    #motion_score = epoch_data["Motion"].mean()
    # Append the motion score to the BrainStateScore data frame
   # BrainStateScore = BrainStateScore.append({"Motion": motion_score}, ignore_index=True)

#%% Estimate a brain state for every 5 second epoch based on motion scores and EEG spectral properties 

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

# Crop data
raw.crop(tmin = 3600 * video_start_index, tmax = 3600 * video_end_index - 1)

# Calculate the number of samples per epoch
samples_per_epoch = int(sample_rate * epoch_length)

# Initialize lists to store the power values
delta_power = []
theta_power = []
delta_theta_ratio = []
wideband_power = []
z_scores = []

# make a folder for all of the images to be saved in, sorted by channel and brain state
os.makedirs(f'{output_dir}/Hypnogram_Data/{current_channel}', exist_ok=True)

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
max_length = max(len(delta_power), len(theta_power), len(delta_theta_ratio), len(wideband_power), len(z_scores), len(motion_scores))

# add Nan to make them the same length
delta_power += [float('nan')] * (max_length - len(delta_power))
theta_power += [float('nan')] * (max_length - len(theta_power))
delta_theta_ratio += [float('nan')] * (max_length - len(delta_theta_ratio))
wideband_power += [float('nan')] * (max_length - len(wideband_power))
z_scores += [float('nan')] * (max_length - len(z_scores))
motion_scores += [float('nan')] * (max_length - len(motion_scores))


# Add the power values to the BrainStateScore DataFrame
BrainStateScore = pd.DataFrame({"Motion": motion_scores})
BrainStateScore["Delta_Power"] = delta_power
BrainStateScore["Theta_Power"] = theta_power
BrainStateScore["Delta_Theta_Ratio"] = delta_theta_ratio
BrainStateScore["Wideband_Power"] = wideband_power
BrainStateScore["Z_Scores"] = z_scores 


#Raises a ValueError when the number of values in the "Motion" 
#column and any of the other columns do not match. 

#if len(BrainStateScore["Motion"]) != len(BrainStateScore["Delta_Power"]):
    #raise ValueError("The number of values in the 'Motion' column and 'Delta_Power' column do not match.")
#if len(BrainStateScore["Motion"]) != len(BrainStateScore["Theta_Power"]):
    #raise ValueError("The number of values in the 'Motion' column and 'Theta_Power' column do not match.")
#if len(BrainStateScore["Motion"]) != len(BrainStateScore["Delta_Theta_Ratio"]):
    #raise ValueError("The number of values in the 'Motion' column and 'Delta_Theta_Ratio' column do not match.")
#if len(BrainStateScore["Motion"]) != len(BrainStateScore["Wideband_Power"]):
    #raise ValueError("The number of values in the 'Motion' column and 'Wideband_Power' column do not match.")
#if len(BrainStateScore["Motion"]) != len(BrainStateScore["Z_Scores"]):
    #raise ValueError("The number of values in the 'Motion' column and 'Z_Scores' column do not match.")


####Brain state scoring
                        

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

Combined_Hypnogram = pd.concat([Hypnogram_1, Hypnogram_2], ignore_index=True, axis = 1)

# Save the DataFrame as a CSV file
Combined_Hypnogram.to_csv(f"{output_dir}/Hypnogram_Data/{current_channel}/{current_channel}_combined_hypnogram.csv")
                          
####Hypnogram 3 - Active versus Inactive

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
    plt.savefig(f'{output_dir}/Hypnogram_Data/{current_channel}/{current_channel}_{state}_EEG_signal.png')
    # Calculate Welch's periodogram
    f, Pxx_den = welch(epoch_data, sample_rate, nperseg=samples_per_epoch)

    # Plot Welch's periodogram
    plt.figure(figsize=(12, 4))
    plt.semilogy(f, Pxx_den)
    plt.title(f"Welch's Periodogram for {state} Brain State")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (V^2/Hz)")
    plt.savefig(f'{output_dir}/Hypnogram_Data/{current_channel}/{current_channel}_{state}_welch_pdg.png')

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
        plt.savefig(f'{output_dir}/Hypnogram_Data/{current_channel}/{current_channel}_{state}_PSD_1.png')

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
        plt.savefig(f'{output_dir}/Hypnogram_Data/{current_channel}/{current_channel}_{state}_PSD_2.png')

#%% Analyze PSD by brain state or activity/inactitvty 
# Load hypnogram and upsample to EEG

############## Analyze PSD by only by Wake, NREM, and REM #######################

#-2 - Unscored
#-1 - Artefact/Movement
# 0 - Wake
# 1 - N1 Sleep
# 2 - N2 Sleep
# 3 - N3 Sleep
# 4 - REM Sleep

hypnogram_1_array = Hypnogram_1['BrainState'].values
hypnogram_2_array = Hypnogram_2['BrainState'].values

#hypno_int = yasa.hypno_str_to_int(hypnogram_1_array, mapping_dict={'': -2, 'REM': 4, 'NREM': 1, 'Resting Wakefulness': 0})

hypno_int_2 = []
for i in hypnogram_1_array:
    if i == '':
        hypno_int_2.append('-1')
    elif i == 'REM':
        hypno_int_2.append('2')
    elif i == 'NREM':
        hypno_int_2.append('1')
    elif i == 'Resting Wakefulness':
        hypno_int_2.append('0')

#yasa.hypno_upsample_to_data(hypno=hypno_30s, sf_hypno=(1/30), data=data, sf_data=sf)
hypno_up = yasa.hypno_upsample_to_data(hypno=hypno_int_2, sf_hypno=(1/5), data=raw, sf_data=100) 

#print(hypno_up.size == raw.shape[1])  # Does the hypnogram have the same number of samples as data?
print(hypno_up.size, 'samples:', hypno_up)

# We use data[0, :] to select only the first channel, which in this case is Cz
#fig = yasa.plot_spectrogram(raw[0, :], 100, hypno)
plt.show()
plt.close()

print(hypno_int_2)
plt.figure(figsize=(20, 6))
plt.plot(hypno_int_2)
plt.title("Sleep Stage Plot")
plt.ylabel("Sleep State")
plt.xlabel("Time (min)")
plt.ylim(bottom=-0.4, top=2.4)
y_ticks = [0, 1, 2]
y_tick_labels = ['Wake', 'NREM', 'REM']
plt.yticks(y_ticks, y_tick_labels)
plt.savefig(f'{output_dir}/Hypnogram_Data/{current_channel}/{current_channel}_sleep_state_plot.png')

plt.figure(figsize=(20, 6))
plt.plot(hypno_int_2[0:100])
plt.title("Sleep Stage Plot")
plt.ylabel("Sleep State")
plt.xlabel("Time (min)")
plt.ylim(bottom=-0.4, top=2.4)
y_ticks = [0, 1, 2]
y_tick_labels = ['Wake', 'NREM', 'REM']
plt.yticks(y_ticks, y_tick_labels)
plt.savefig(f'{output_dir}/Hypnogram_Data/{current_channel}/{current_channel}0_100_sleep_state_plot.png')

plt.figure(figsize=(20, 6))
plt.plot(hypno_int_2[100:200])
plt.title("Sleep Stage Plot")
plt.ylabel("Sleep State")
plt.xlabel("Time (min)")
plt.ylim(bottom=-0.4, top=2.4)
y_ticks = [0, 1, 2]
y_tick_labels = ['Wake', 'NREM', 'REM']
plt.yticks(y_ticks, y_tick_labels)
plt.savefig(f'{output_dir}/Hypnogram_Data/{current_channel}/{current_channel}100_200_sleep_state_plot.png')

plt.figure(figsize=(20, 6))
plt.plot(hypno_int_2[300:400])
plt.title("Sleep Stage Plot")
plt.ylabel("Sleep State")
plt.xlabel("Time (min)")
plt.ylim(bottom=-0.4, top=2.4)
y_ticks = [0, 1, 2]
y_tick_labels = ['Wake', 'NREM', 'REM']
plt.yticks(y_ticks, y_tick_labels)
plt.savefig(f'{output_dir}/Hypnogram_Data/{current_channel}/{current_channel}300_400_sleep_state_plot.png')


yasa.bandpower(raw, include=(2))


############## Analyze PSD by motion or no motion  #########################

hypno_active = yasa.hypno_str_to_int(hypnogram_2_array, mapping_dict={"Wake": 0, "NREM": 1, "REM": 1})
#hypno_active2 = yasa.hypno_consolidate_stages(hypno_int_2, 3, 2)


#%% Use YASA to calculate sleep state statistics 
#sleep_statistics(hypno_int_2, sf_hyp=1/30)
    # Time in Bed (TIB): total duration of the hypnogram.
    # Sleep Period Time (SPT): duration from first to last period of sleep.
    # Wake After Sleep Onset (WASO): duration of wake periods within SPT.
    # Total Sleep Time (TST): total duration of N1 + N2 + N3 + REM sleep in SPT.
    # Sleep Efficiency (SE): TST / TIB * 100 (%).
    # Sleep Maintenance Efficiency (SME): TST / SPT * 100 (%).
    # W, N1, N2, N3 and REM: sleep stages duration. NREM = N1 + N2 + N3.
    # (W, … REM): sleep stages duration expressed in percentages of TST.
    # Latencies: latencies of sleep stages from the beginning of the record.
    # Sleep Onset Latency (SOL): Latency to first epoch of any sleep

# Calculate the sleep stages transition matrix, probs is the probability transition matrix, 
#i.e. given that the current sleep stage is A, what is the probability that the next sleep stage is B. 

counts, probs = transition_matrix(hypno_int_2)
probs.round(3)

# Stability of sleep is calculated by taking the average of the diagonal values of N2, N3 and REM sleep:
np.diag(probs.loc[2:, 2:]).mean().round(3)

# Plot
grid_kws = {"height_ratios": (.9, .05), "hspace": .1}
f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws,
                                figsize=(5, 5))
sns.heatmap(probs, ax=ax, square=False, vmin=0, vmax=1, cbar=True,
            cbar_ax=cbar_ax, cmap='YlOrRd', annot=True, fmt='.2f',
            cbar_kws={"orientation": "horizontal", "fraction": 0.1,
                      "label": "Transition probability"})

ax.set_xlabel("To sleep stage")
ax.xaxis.tick_top()
ax.set_ylabel("From sleep stage")
ax.xaxis.set_label_position('top')
plt.savefig(f'{output_dir}/Hypnogram_Data/{current_channel}/{current_channel}_sleep_stage_trn_matrix.png')
