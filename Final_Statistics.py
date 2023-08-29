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

# Check if metadata file exists
input_dir = sys.argv[1]
print(input_dir)
output_dir = sys.argv[2]
print(output_dir)
batch_id = sys.argv[3]
print(batch_id)

metadata_file = input_dir + "/" + batch_id + "_metadata.csv"
if os.path.isfile(metadata_file):
    metadata = pd.read_csv(metadata_file)
    print("Metadata file exists.")
else:
    error = "Designated metadata file does not exist! exiting..."
    sys.exit(error)

os.makedirs(f'{output_dir}/Final_Output', exist_ok=True)

# call values from metadata file
detect_spikes = metadata.loc[metadata['Variable'] == 'spikes']['Value'].tolist()[0]
detect_swds = metadata.loc[metadata['Variable'] == 'swds']['Value'].tolist()[0]
detect_seizures = metadata.loc[metadata['Variable'] == 'seizures']['Value'].tolist()[0]
detect_spindles = metadata.loc[metadata['Variable'] == 'spindles']['Value'].tolist()[0]
detect_threshold_spikes = metadata.loc[metadata['Variable'] == 'Threshold Spikes']['Value'].tolist()[0]
detect_rolling_spikes = metadata.loc[metadata['Variable'] == 'Rolling Spikes']['Value'].tolist()[0]
video_exists = metadata.loc[metadata['Variable'] == 'Video File']['Value'].tolist()[0]

channel_ids = ast.literal_eval(metadata.loc[metadata['Variable'] == 'Channel Names']['Value'].tolist()[0])
group = ast.literal_eval(metadata.loc[metadata['Variable'] == 'Group']['Value'].tolist()[0])


# repetitive functions to get statistics
def calculate_average(column):
    numerical_values = [float(value) for value in column if value.replace('.', '', 1).isdigit()]
    if numerical_values:
        return sum(numerical_values) / len(numerical_values)
    else:
        return 0

# function to get event counts -- just counts the number of rows in the event csv file and subtracts by 1
def number_of_events(file):
    try:
        with open(file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            row_count = sum(1 for row in csv_reader)
            print(int(row_count) - 1)
            return int(row_count) - 1
    except FileNotFoundError:
        return 0

# gets average value of a specific event
def column_average(file, column):
    try:
        with open(file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            data = list(csv_reader)
            column_info = [row[column - 1] for row in data if len(row) >= column]
            average = calculate_average(column_info)
            print(average)
            return average
    except FileNotFoundError:
        return 0

# find percent of spikes that are labelled as Seizure
def PercentSeizureSpikes(file):
    seizure_spike_count = 0
    try:
        with open(file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            row_count = sum(1 for row in csv_reader)
            csv_file.seek(0)
            for row in csv_reader:
                if row and row[2] == 'Seizure':
                    seizure_spike_count += 1
            percent = 100 * (seizure_spike_count / row_count)
            return percent
    except FileNotFoundError:
        return 0

# finds proportion of time in each brain state
def getTimeinState(file, state, other_state):
    state_count = 0
    other_state_count = 0
    try:
        with open(file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            row_count = sum(1 for row in csv_reader)
            csv_file.seek(0)
            for row in csv_reader:
                if row and row[2] == state:
                    state_count += 1
            for row in csv_reader:
                if row and row[2] == other_state:
                    other_state_count += 1
            if state_count == 0 and other_state_count == 0:
                percent_state = 0
            else:
                percent_state = 100 * (state_count / (state_count + other_state_count))
            return percent_state
    except FileNotFoundError:
        return 0

# sets notable statistic names
csv_filename = f'{output_dir}/Final_Output/{batch_id}_final_output.csv'
variables = [
        "Batch_ID", "Channel", "Group", "Seizure Count", "Average Seizure Spikes",
        "Percent Seizure Spikes Threshold", "Percent Seizure Spikes Rolling",
        "Threshold Spike Count", "Average Threshold Amplitude",
        "Rolling Spike Count", "Average Rolling Amplitude",
        "SWD Count", "Average SWD Length",
        "Spindle Count", "Average Spindle Length",
        "Percent Time Awake", "Percent Time Asleep"
    ]

# create csv file with statistic names as headers
with open(csv_filename, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(variables)    

# in each channel, appends data
for current_channel in channel_ids:

    index = channel_ids.index(current_channel)
    group_number = group[index]

    if detect_seizures == 'True':

        seizure_file = f"{output_dir}/Event_Detection_Data/Seizures/{current_channel}_Seizures.csv"
        # count number of seizures
        seizure_count = number_of_events(seizure_file)
        seizure_count

        # count average length of seizure by spikes
        column = 3
        avg_seizure_spikes = column_average(seizure_file, column)
        avg_seizure_spikes
    # if no seizure detection, replace w nan values
    else:
        seizure_count = np.nan
        avg_seizure_spikes = np.nan

# find spike amplitude for both rolling and threshold spikes
    if detect_spikes == 'True':
        if detect_threshold_spikes == 'True':
            threshold_spikes_file = f"{output_dir}/Event_Detection_Data/Spikes/Threshold_Spikes/{current_channel}/{current_channel}_threshold_spikes.csv"
            threshold_spike_count = number_of_events(threshold_spikes_file)
            threshold_spike_count

            average_threshold_amplitude = column_average(threshold_spikes_file, 2)
            average_threshold_amplitude
            
            # if detect seizures, also find percent seizure spikes
            if detect_seizures == 'True':
                Percent_Seizure_Spikes_T_file = f"{output_dir}/Event_Detection_Data/Spikes/Threshold_Spikes/{current_channel}/{current_channel}_threshold_spikes_with_seizures.csv"
                Percent_Seizure_Spikes_T = PercentSeizureSpikes(Percent_Seizure_Spikes_T_file)
                Percent_Seizure_Spikes_T

        else:
            threshold_spike_count = np.nan
            average_threshold_amplitude = np.nan
            Percent_Seizure_Spikes_T = np.nan

        if detect_rolling_spikes == 'True':
            rolling_spikes_file = f"{output_dir}/Event_Detection_Data/Spikes/Rolling_Spikes/{current_channel}/{current_channel}_rolling_spikes.csv"
            rolling_spike_count = number_of_events(rolling_spikes_file)
            rolling_spike_count

            average_rolling_amplitude = column_average(rolling_spikes_file, 2)
            average_rolling_amplitude

            # if detect seizures, also find percent seizure spikes
            if detect_seizures == 'True':
                Percent_Seizure_Spikes_R_file = f"{output_dir}/Event_Detection_Data/Spikes/Rolling_Spikes/{current_channel}/{current_channel}_rolling_spikes_with_seizures.csv"
                Percent_Seizure_Spikes_R = PercentSeizureSpikes(Percent_Seizure_Spikes_R_file)
                Percent_Seizure_Spikes_R
        else:
            rolling_spike_count = np.nan
            average_rolling_amplitude = np.nan
            Percent_Seizure_Spikes_R = np.nan
        

                
# find relevant SWD statistics; set nan if not set to find them
    if detect_swds == 'True':
        swds_file = f"{output_dir}/Event_Detection_Data/SWDs/{current_channel}_SWD_events.csv"
        swd_count = number_of_events(swds_file)
        swd_count

        average_swd_length = column_average(swds_file, 5)
        average_swd_length
    else:
        swd_count = np.nan
        average_swd_length = np.nan

# find relevant spindle statistics; set nan if not set to find them
    if detect_spindles == 'True':
        spindles_file = f"{output_dir}/Event_Detection_Data/Spindles/{current_channel}_Spindle_events.csv"
        spindle_count = number_of_events(spindles_file)
        spindle_count

        average_spindle_length = column_average(spindles_file, 5)
        average_spindle_length
    else:
        spindle_count = np.nan
        average_spindle_length = np.nan
        
    if video_exists != 'NO VIDEO':
        file = f"{output_dir}/Hypnogram_Data/{current_channel}/{current_channel}_combined_hypnogram.csv"
        sleep_time = getTimeinState(file, 'Sleep', 'Wake')
        sleep_time
        wake_time = getTimeinState(file, 'Wake', 'Sleep')
        wake_time
    else:
        sleep_time = np.nan
        wake_time = np.nan

# set variable values and plot them in the next available row for the specified channel
    values = [
        batch_id, current_channel, group_number, seizure_count, avg_seizure_spikes,
        Percent_Seizure_Spikes_T, Percent_Seizure_Spikes_R,
        threshold_spike_count, average_threshold_amplitude,
        rolling_spike_count, average_rolling_amplitude,
        swd_count, average_swd_length,
        spindle_count, average_spindle_length,
        wake_time, sleep_time
    ]
    
    with open(csv_filename, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(values)
