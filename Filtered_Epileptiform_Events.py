import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import csv

# Check if metadata file exists
input_dir = sys.argv[1]
print(input_dir)
output_dir = sys.argv[2]
print(output_dir)
batch_id = sys.argv[3]
print(batch_id)
current_channel = sys.argv[4]
print(current_channel)

# call metadata file
metadata_file = input_dir + "/" + batch_id + "_metadata.csv"
if os.path.isfile(metadata_file):
    metadata = pd.read_csv(metadata_file)
    print("Metadata file exists.")
else:
    error = "Designated metadata file does not exist! exiting..."
    sys.exit(error)

# call values from metadata file
sample_rate = float(metadata.loc[metadata['Variable'] == 'Sample Frequency']['Value'].tolist()[0])
epoch_length = int(metadata.loc[metadata['Variable'] == 'Epoch Time']['Value'].tolist()[0])
detect_threshold_spikes = metadata.loc[metadata['Variable'] == 'Threshold Spikes']['Value'].tolist()[0]
detect_rolling_spikes = metadata.loc[metadata['Variable'] == 'Rolling Spikes']['Value'].tolist()[0]
    
# Calculate the number of samples per epoch
samples_per_epoch = int(sample_rate * epoch_length)

# Read combination hypnogram csv for current channel
brainstates_df = pd.read_csv(f"{output_dir}/Hypnogram_Data/{current_channel}/{current_channel}_combined_hypnogram.csv") 

# set brain state and wake state values 
brain_state_array = brainstates_df['0']
wake_state_array = brainstates_df['1']

# Read Spike_Data csv for current channel
if detect_threshold_spikes == 'True':
    spike_data_df = pd.read_csv(f"{output_dir}/Event_Detection_Data/Spikes/Threshold_Spikes/{current_channel}/{current_channel}_threshold_spikes.csv")
    
    
        # Get the spike times and turn them into an array
    spike_data_array = (spike_data_df['Spike Times'].values / 1000)
    

    # Make a new array that divides by sf * epoch time to get the epoch number
    epoch_number_array = np.floor(spike_data_df['Spike Times'].values / samples_per_epoch)

    # Define the range intervals with the last interval of 5 included
    if len(epoch_number_array) > 0:
        max_value = np.ceil(max(epoch_number_array)) + 1
    else:
        max_value = 1
    intervals = np.arange(0, max_value + 1, 1)

    # count occurrences in each interval
    counts = np.histogram(epoch_number_array, bins=intervals)[0]

    # Display the counts for each interval
    spikes_per_epoch = []
    for i in range(len(intervals) - 1):
        lower_bound = intervals[i]
        upper_bound = intervals[i + 1]
        spikes_per_epoch.append(counts[i])

    max_length = max(len(epoch_number_array), len(spikes_per_epoch), len(wake_state_array), len(brain_state_array))

    # create series
    epoch_number_series = pd.Series(epoch_number_array, index=np.arange(1, len(epoch_number_array) + 1))
    spikes_per_epoch_series = pd.Series(spikes_per_epoch, index=np.arange(1, len(spikes_per_epoch) + 1))
    wake_state_series = pd.Series(wake_state_array, index=np.arange(1, len(wake_state_array) + 1))
    brain_state_series = pd.Series(brain_state_array, index=np.arange(1, len(brain_state_array) + 1))
    spike_data_series = pd.Series(spike_data_array, index=np.arange(1, len(spike_data_array) + 1))


    # check if the lengths of epoch_number_series, wake_state_series, and brain_state_series are the same
    if len(epoch_number_series) != len(wake_state_series) or len(epoch_number_series) != len(brain_state_series):
        epoch_number_series = epoch_number_series.reindex(range(max_length), fill_value=np.nan)
        wake_state_series = wake_state_series.reindex(range(max_length), fill_value=np.nan)
        brain_state_series = brain_state_series.reindex(range(max_length), fill_value=np.nan)
        spike_data_series = spike_data_series.reindex(range(max_length), fill_value=np.nan)

    # do the same for spikes_per_epoch_series (necessary for silly code later)
    if len(spikes_per_epoch_series) != len(wake_state_series):
        spikes_per_epoch_series = spikes_per_epoch_series.reindex(range(max_length), fill_value=np.nan)

    # create the new_wake_state list
    new_wake_state = []
    for i in epoch_number_series:
        if not pd.isnull(i) and int(i) < len(wake_state_series):
            new_wake_state.append(wake_state_series.iloc[int(i)])
        else:
            new_wake_state.append(np.nan)

    # create the new_brain_state list
    new_brain_state = []
    for i in epoch_number_series:
        if not pd.isnull(i) and int(i) < len(brain_state_series):
            new_brain_state.append(brain_state_series.iloc[int(i)])
        else:
            new_brain_state.append(np.nan)

    # create a DataFrame with the necessary series
    spikes_by_state = pd.DataFrame({
        'Spike Time (s)': spike_data_series,
        'Wake State': new_wake_state,
        'Brain State': new_brain_state
    })


    # create another dataframe for epochs and spikes in each epoch
    spikes_by_epoch = pd.DataFrame({'Spikes per Epoch': spikes_per_epoch_series})


    # create folder to save csv in
    os.makedirs(f'{output_dir}/Filtered_Epileptiform_Events/Threshold_Spikes/{current_channel}', exist_ok=True)

    spikes_by_state.to_csv(f"{output_dir}/Filtered_Epileptiform_Events/Threshold_Spikes/{current_channel}/{current_channel}_spikes_by_state.csv")
    spikes_by_epoch.to_csv(f"{output_dir}/Filtered_Epileptiform_Events/Threshold_Spikes/{current_channel}/{current_channel}_spikes_by_epoch.csv")


    # time to count amount of spikes per state

    #first we do it for wake state
    # dictionary to store the sum for each unique value in wake state
    wake_state_sum = {}

    spike_data_series = spike_data_series.reindex(range(max_length), fill_value=np.nan)
    wake_state_series = wake_state_series.reindex(range(max_length), fill_value=np.nan)
    brain_state_series = brain_state_series.reindex(range(max_length), fill_value=np.nan)
    spikes_per_epoch_series = spikes_per_epoch_series.reindex(range(max_length), fill_value=np.nan)

    new_wake_state = []
    for i in epoch_number_series:
        if not pd.isnull(i) and int(i) < len(wake_state_series):
            new_wake_state.append(wake_state_series.iloc[int(i)])
        else:
            new_wake_state.append(np.nan)

    # create the new_brain_state list
    new_brain_state = []
    for i in epoch_number_series:
        if not pd.isnull(i) and int(i) < len(brain_state_series):
            new_brain_state.append(brain_state_series.iloc[int(i)])
        else:
            new_brain_state.append(np.nan)

    events_by_state = pd.DataFrame({
        'Spikes per Epoch': spikes_per_epoch_series,
        'Spike Time (s)': spike_data_series,
        'Wake State': new_wake_state,
        'Brain State': new_brain_state})

    # calculate sums and ignore Nans
    for index, row in events_by_state.iterrows():
        spike_epoch = row['Spikes per Epoch']
        state_wake = row['Wake State']
        if pd.notnull(spike_epoch):  # Check if the 'Spikes per Epoch' value is not NaN
            wake_state_sum[state_wake] = wake_state_sum.get(state_wake, 0) + spike_epoch

    # display sums for each unique value and make sure they're not Nan
    for key, value in wake_state_sum.items():
        if pd.notnull(key):
            print(f"Sum for '{key}': {value}")

    # write to csv file made before
    with open(f"{output_dir}/Filtered_Epileptiform_Events/Threshold_Spikes/{current_channel}/{current_channel}_wake_spikes_filtered_events.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Wake State', 'Number of Spikes per State'])
        for key, value in wake_state_sum.items():
            if pd.notnull(key):  # Check if the key is not NaN
                writer.writerow([key, value])

    # same code but for brain state

    # dictionary
    sleep_state_sum = {}

    # calculate sums and ignore Nans
    for index, row in events_by_state.iterrows():
        spike_epoch = row['Spikes per Epoch']
        state_brain = row['Brain State']
        if pd.notnull(spike_epoch):
            sleep_state_sum[state_brain] = sleep_state_sum.get(state_brain, 0) + spike_epoch

    # display sums for each unique value and make sure they're not Nan
    for key, value in sleep_state_sum.items():
        if pd.notnull(key):
            print(f"Sum for '{key}': {value}")

    # write to csv
    with open(f"{output_dir}/Filtered_Epileptiform_Events/Threshold_Spikes/{current_channel}/{current_channel}_sleep_spikes_filtered_events.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sleep State', 'Number of Spikes per State'])
        for key, value in sleep_state_sum.items():
            if pd.notnull(key):
                writer.writerow([key, value])

#########################
#########################
#########################
#########################
#########################

if detect_rolling_spikes == 'True':
    spike_data_df = pd.read_csv(f"{output_dir}/Event_Detection_Data/Spikes/Rolling_Spikes/{current_channel}/{current_channel}_rolling_spikes.csv")

        # Get the spike times and turn them into an array
    spike_data_array = (spike_data_df['Spike Times'].values / 1000)

     # Make a new array that divides by sf * epoch time to get the epoch number
    epoch_number_array = np.floor(spike_data_df['Spike Times'].values / samples_per_epoch)

    # Define the range intervals with the last interval of 5 included
    if len(epoch_number_array) > 0:
        max_value = np.ceil(max(epoch_number_array)) + 1
    else:
        max_value = 1
    intervals = np.arange(0, max_value + 1, 1)

    # count occurrences in each interval
    counts = np.histogram(epoch_number_array, bins=intervals)[0]

    # Display the counts for each interval
    spikes_per_epoch = []
    for i in range(len(intervals) - 1):
        lower_bound = intervals[i]
        upper_bound = intervals[i + 1]
        spikes_per_epoch.append(counts[i])

    max_length = max(len(epoch_number_array), len(spikes_per_epoch), len(wake_state_array), len(brain_state_array))

    # create series
    epoch_number_series = pd.Series(epoch_number_array, index=np.arange(1, len(epoch_number_array) + 1))
    spikes_per_epoch_series = pd.Series(spikes_per_epoch, index=np.arange(1, len(spikes_per_epoch) + 1))
    wake_state_series = pd.Series(wake_state_array, index=np.arange(1, len(wake_state_array) + 1))
    brain_state_series = pd.Series(brain_state_array, index=np.arange(1, len(brain_state_array) + 1))
    spike_data_series = pd.Series(spike_data_array, index=np.arange(1, len(spike_data_array) + 1))


    # check if the lengths of epoch_number_series, wake_state_series, and brain_state_series are the same
    if len(epoch_number_series) != len(wake_state_series) or len(epoch_number_series) != len(brain_state_series):
        epoch_number_series = epoch_number_series.reindex(range(max_length), fill_value=np.nan)
        wake_state_series = wake_state_series.reindex(range(max_length), fill_value=np.nan)
        brain_state_series = brain_state_series.reindex(range(max_length), fill_value=np.nan)
        spike_data_series = spike_data_series.reindex(range(max_length), fill_value=np.nan)

    # do the same for spikes_per_epoch_series (necessary for silly code later)
    if len(spikes_per_epoch_series) != len(wake_state_series):
        spikes_per_epoch_series = spikes_per_epoch_series.reindex(range(max_length), fill_value=np.nan)

    # create the new_wake_state list
    new_wake_state = []
    for i in epoch_number_series:
        if not pd.isnull(i) and int(i) < len(wake_state_series):
            new_wake_state.append(wake_state_series.iloc[int(i)])
        else:
            new_wake_state.append(np.nan)

    # create the new_brain_state list
    new_brain_state = []
    for i in epoch_number_series:
        if not pd.isnull(i) and int(i) < len(brain_state_series):
            new_brain_state.append(brain_state_series.iloc[int(i)])
        else:
            new_brain_state.append(np.nan)

    # create a DataFrame with the necessary series
    spikes_by_state = pd.DataFrame({
        'Spike Time (s)': spike_data_series,
        'Wake State': new_wake_state,
        'Brain State': new_brain_state
    })


    # create another dataframe for epochs and spikes in each epoch
    spikes_by_epoch = pd.DataFrame({'Spikes per Epoch': spikes_per_epoch_series})


    # create folder to save csv in
    os.makedirs(f'{output_dir}/Filtered_Epileptiform_Events/Rolling_Spikes/{current_channel}', exist_ok=True)

    spikes_by_state.to_csv(f"{output_dir}/Filtered_Epileptiform_Events/Rolling_Spikes/{current_channel}/{current_channel}_spikes_by_state.csv")
    spikes_by_epoch.to_csv(f"{output_dir}/Filtered_Epileptiform_Events/Rolling_Spikes/{current_channel}/{current_channel}_spikes_by_epoch.csv")


    # time to count amount of spikes per state

    #first we do it for wake state
    # dictionary to store the sum for each unique value in wake state
    wake_state_sum = {}

    spike_data_series = spike_data_series.reindex(range(max_length), fill_value=np.nan)
    wake_state_series = wake_state_series.reindex(range(max_length), fill_value=np.nan)
    brain_state_series = brain_state_series.reindex(range(max_length), fill_value=np.nan)
    spikes_per_epoch_series = spikes_per_epoch_series.reindex(range(max_length), fill_value=np.nan)

    new_wake_state = []
    for i in epoch_number_series:
        if not pd.isnull(i) and int(i) < len(wake_state_series):
            new_wake_state.append(wake_state_series.iloc[int(i)])
        else:
            new_wake_state.append(np.nan)

    # create the new_brain_state list
    new_brain_state = []
    for i in epoch_number_series:
        if not pd.isnull(i) and int(i) < len(brain_state_series):
            new_brain_state.append(brain_state_series.iloc[int(i)])
        else:
            new_brain_state.append(np.nan)

    events_by_state = pd.DataFrame({
        'Spikes per Epoch': spikes_per_epoch_series,
        'Spike Time (s)': spike_data_series,
        'Wake State': new_wake_state,
        'Brain State': new_brain_state})

    # calculate sums and ignore Nans
    for index, row in events_by_state.iterrows():
        spike_epoch = row['Spikes per Epoch']
        state_wake = row['Wake State']
        if pd.notnull(spike_epoch):  # Check if the 'Spikes per Epoch' value is not NaN
            wake_state_sum[state_wake] = wake_state_sum.get(state_wake, 0) + spike_epoch

    # display sums for each unique value and make sure they're not Nan
    for key, value in wake_state_sum.items():
        if pd.notnull(key):
            print(f"Sum for '{key}': {value}")

    # write to csv file made before
    with open(f"{output_dir}/Filtered_Epileptiform_Events/Rolling_Spikes/{current_channel}/{current_channel}_wake_spikes_filtered_events.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Wake State', 'Number of Spikes per State'])
        for key, value in wake_state_sum.items():
            if pd.notnull(key):  # Check if the key is not NaN
                writer.writerow([key, value])

    # same code but for brain state

    # dictionary
    sleep_state_sum = {}

    # calculate sums and ignore Nans
    for index, row in events_by_state.iterrows():
        spike_epoch = row['Spikes per Epoch']
        state_brain = row['Brain State']
        if pd.notnull(spike_epoch):
            sleep_state_sum[state_brain] = sleep_state_sum.get(state_brain, 0) + spike_epoch

    # display sums for each unique value and make sure they're not Nan
    for key, value in sleep_state_sum.items():
        if pd.notnull(key):
            print(f"Sum for '{key}': {value}")

    # write to csv
    with open(f"{output_dir}/Filtered_Epileptiform_Events/Rolling_Spikes/{current_channel}/{current_channel}_sleep_spikes_filtered_events.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sleep State', 'Number of Spikes per State'])
        for key, value in sleep_state_sum.items():
            if pd.notnull(key):
                writer.writerow([key, value])

                
os.makedirs(f'{output_dir}/Filtered_Epileptiform_Events/completion_messages', exist_ok=True)
data = {'Message': [f'filtered epileptiform events complete for {current_channel}']}
df = pd.DataFrame(data)
df.to_csv(f"{output_dir}/Filtered_Epileptiform_Events/completion_messages/{current_channel}_completion.csv")