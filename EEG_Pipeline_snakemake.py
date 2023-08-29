# -------------------------------------------------------------------------------------------
# EEG Pipeline Snakemake Script:
# -------------------------------------------------------------------------------------------

# Authors: Cathrine Petersen, Melanie Das, Syon Mansur

# Description: This pipeline enables motion detection, spectral analysis, hypnogram generation,
# epileptiform event detection, and output of consolidated statistics.

import os
import sys
import numpy as np
import pandas as pd
import snakemake.io
import glob
import ast

# ---------------------------------------------------------------------------
# Set up
# ---------------------------------------------------------------------------

# Define config variables
in_dir = config["in_dir"]
out_dir = config["out_dir"]
temp_dir = config["temp_dir"]
scripts_dir = config["scripts"]
batch_id = config["batch_id"]

# Check if metadata file exists
metadata_file = in_dir + "/" + batch_id + "_metadata.csv"
if os.path.isfile(metadata_file):
    metadata = pd.read_csv(metadata_file)
    print("Metadata file exists.")
else:
    error = "Designated metadata file does not exist! exiting..."
    sys.exit(error)
    
# Import values from metadata
channel_ids = ast.literal_eval(metadata.loc[metadata['Variable'] == 'Channel Names']['Value'].tolist()[0])
mouse_ids = ast.literal_eval(metadata.loc[metadata['Variable'] == 'Mouse IDs']['Value'].tolist()[0])

# more variables from the metadata -- these are used to determine which inputs and outputs to use (event detection)
detect_spikes = metadata.loc[metadata['Variable'] == 'spikes']['Value'].tolist()[0]
detect_swds = metadata.loc[metadata['Variable'] == 'swds']['Value'].tolist()[0]
detect_seizures = metadata.loc[metadata['Variable'] == 'seizures']['Value'].tolist()[0]
detect_spindles = metadata.loc[metadata['Variable'] == 'spindles']['Value'].tolist()[0]
detect_threshold_spikes = metadata.loc[metadata['Variable'] == 'Threshold Spikes']['Value'].tolist()[0]
detect_rolling_spikes = metadata.loc[metadata['Variable'] == 'Rolling Spikes']['Value'].tolist()[0]
video_exists = metadata.loc[metadata['Variable'] == 'Video File']['Value'].tolist()[0]

# sets video file if metadata file indicates that the user selected analysis with video
if video_exists != 'NO VIDEO':
    video_file = in_dir + "/Raw_EEG_Videos/" + batch_id + ".wmv"
    print("Video file path: ", video_file)

    # Check if video file exists
    if os.path.isfile(video_file):
        print("Video file exists.")
    else:
        print("No video file present.")

# set eeg_file (as specified in setup script
eeg_file = in_dir + "/Raw_EEG_Files/" + batch_id + ".edf"

# motion_scores = in_dir + "/motion_test.csv"

print("-----------------------------------")
print("Input directory: ", in_dir)
print("Output directory: ", out_dir)
print("Temporary directory: ", temp_dir)
print("Scripts directory: ", scripts_dir)
print("Metadata file path: ", metadata_file)
print("-----------------------------------")
print("Batch ID: ", batch_id)
print("EEG file path: ", eeg_file)
print("-----------------------------------")

# Check if EEG file exists
if os.path.isfile(eeg_file):
    print("EEG file exists: ", eeg_file)
else:
    print("EEG file does NOT exist: ", eeg_file)
    error = "A designated EEG file does not exist! exiting..."
    sys.exit(error)

# ---------------------------------------------------------------------------
# Snakemake rules
# ---------------------------------------------------------------------------

localrules: all

# create a list to append to for all of the rule all inputs (snakemake outputs)
event_detection_outputs = list()

# if user indicated they wanted to detect spikes, add that to the rule all inputs
if detect_spikes == 'True':
    # check if user wanted threshold or rolling spikes, change input accordingly
    if detect_threshold_spikes == 'True':
        event_detection_outputs.append(expand("{output_dir}/Event_Detection_Data/Spikes/Threshold_Spikes/{chan_ids}/{chan_ids}_threshold_spikes.csv", 
                                           output_dir = out_dir, chan_ids = channel_ids))
    if detect_rolling_spikes == 'True':
        event_detection_outputs.append(expand("{output_dir}/Event_Detection_Data/Spikes/Rolling_Spikes/{chan_ids}/{chan_ids}_rolling_spikes.csv", 
                                           output_dir = out_dir, chan_ids = channel_ids))

# if user indicated they wanted to detect swds, add that to the rule all inputs        
if detect_swds == 'True':
    event_detection_outputs.append(expand("{output_dir}/Event_Detection_Data/SWDs/{chan_ids}_SWD_events.csv", 
                                           output_dir = out_dir, chan_ids = channel_ids))
    
# if user indicated they wanted to detect seizures, add that to the rule all inputs        
if detect_seizures == 'True':
    event_detection_outputs.append(expand("{output_dir}/Event_Detection_Data/Seizures/{chan_ids}_Seizures.csv", 
                                           output_dir = out_dir, chan_ids = channel_ids))
    
# if user indicated they wanted to detect spindles, add that to the rule all inputs        
if detect_spindles == 'True':
    event_detection_outputs.append(expand("{output_dir}/Event_Detection_Data/Spindles/{chan_ids}_Spindle_events.csv", 
                                           output_dir = out_dir, chan_ids = channel_ids))
    
# if user indicated video analysis, add all motion detection outputs to the that to the rule all inputs        
if video_exists != 'NO VIDEO':
    #event_detection_outputs.append(out_dir + "/Motion_Tracking/Motion_Intermediates/" + batch_id + "_frame_rate.txt")
    #event_detection_outputs.append(f"{out_dir}/Motion_Tracking/Motion_Intermediates/{batch_id}_background_subtraction_complete.csv")
    #event_detection_outputs.append(f"{out_dir}/Motion_Tracking/Motion_Intermediates/{batch_id}_avg_pixels_plot.png")
    event_detection_outputs.append(expand("{output_dir}/Hypnogram_Data/{chan_ids}/{chan_ids}_combined_hypnogram.csv",
                                           output_dir = out_dir, chan_ids = channel_ids))
    event_detection_outputs.append(expand("{output_dir}/Filtered_Epileptiform_Events/completion_messages/{chan_ids}_completion.csv",
               output_dir = out_dir, chan_ids = channel_ids))
    event_detection_outputs.append(expand("{output_dir}/Motion_Tracking/Motion_Outputs/{chan_ids}/{chan_ids}_motion.csv", 
               output_dir = out_dir, chan_ids = mouse_ids))

# if the user indicated any desire for event detection, add finemapped spectral analysis outputs to the rule all inputs and the final output detection
if detect_spikes == 'True' or detect_seizures == 'True' or detect_spindles == 'True' or detect_swds == 'True':
    event_detection_outputs.append(out_dir + "/Final_Output/" + batch_id + "_final_output.csv")
    if video_exists != 'NO VIDEO':
        event_detection_outputs.append(expand("{output_dir}/Fine_Mapped_Spectral_Analysis/{chan_ids}/{chan_ids}_full_Fine_Mapped_Spectral_Analysis.png",
                   output_dir = out_dir, chan_ids = channel_ids))
    
rule all:
    input:
        event_detection = event_detection_outputs,
    output:
        final = "PipelineCompletion.txt"
    threads: 1
    resources:
        mem_qsub = "2G",
        job_time = "00:15:00",
    shell:
        "echo Pipeline created by Cathrine Petersen, Melanie Das and Syon Mansur.; "
        "echo Completed Successfully at `date` | tee {output.final}; "

# ---------------------------------------------------------------------------
# Motion tracking
# ---------------------------------------------------------------------------

# if the video exists, run the extractFrameRate, getBackground, subtractBackground, detectMotion, AND generateHypnogram rules
if video_exists != 'NO VIDEO':
    rule extractFrameRate:
        input:
            video_file,
            metadata_file,
        output:
            frame_rate = out_dir + "/Motion_Tracking/Motion_Intermediates/" + batch_id + "_frame_rate.txt",
        threads: 1
        resources:
            mem_qsub = "50G",
            job_time = "05:00:00",
        shell:
            "ffprobe -fflags +genpts -i {video_file} -select_streams v -show_entries frame=coded_picture_number,pkt_pts_time -of csv=p=0:nk=1 -v 0 | tee {out_dir}/Motion_Tracking/Motion_Intermediates/{batch_id}_frame_rate.txt"

    rule getBackground:
        input:
            video_file, 
            metadata_file,
            frame_rate = out_dir + "/Motion_Tracking/Motion_Intermediates/" + batch_id + "_frame_rate.txt",
        output:
            expand("{output_dir}/Motion_Tracking/Motion_Intermediates/{batch}_segment_data.csv", 
                   output_dir = out_dir, batch = batch_id), 
        threads: 1
        resources:
            mem_qsub = "50G",
            job_time = "15:00:00",
        shell:
            "python3 {scripts_dir}/EEG_Pipeline_get_background_frames.py {in_dir} {out_dir} {batch_id} {video_file} {metadata_file}"


    rule subtractBackground:
        input:
            video_file,
            metadata_file,
            segment_data = out_dir + "/Motion_Tracking/Motion_Intermediates/" + batch_id + "_segment_data.csv",
        output:
            completion_check = out_dir + "/Motion_Tracking/Motion_Intermediates/" + batch_id + "_background_subtraction_complete.csv",
        threads: 1
        resources:
            mem_qsub = "50G",
            job_time = "15:00:00",
        shell:
            "python3 {scripts_dir}/EEG_pipeline_subtract_background.py {in_dir} {out_dir} {batch_id} {video_file} {metadata_file}"

    rule detectMotion:
        input:
            completion_check = out_dir + "/Motion_Tracking/Motion_Intermediates/" + batch_id + "_background_subtraction_complete.csv",
            frame_rate = out_dir + "/Motion_Tracking/Motion_Intermediates/" + batch_id + "_frame_rate.txt",
            segment_data = out_dir + "/Motion_Tracking/Motion_Intermediates/" + batch_id + "_segment_data.csv",
        output:
            motion_csv = out_dir + "/Motion_Tracking/Motion_Outputs/{mouse_id}/{mouse_id}_motion.csv",
            threshold_csv = out_dir + "/Motion_Tracking/Motion_Outputs/{mouse_id}/{mouse_id}_motion_threshold_parameters.csv",
        threads: 1
        resources:
            mem_qsub = "30G",
            job_time = "04:00:00",
        shell:
            "python3 {scripts_dir}/EEG_Pipeline_detect_motion.py {in_dir} {out_dir} {batch_id} {wildcards.mouse_id} {metadata_file}"



# ---------------------------------------------------------------------------
# Hypnogram generation (INCLUDED IN THE IF STATEMENT ABOVE)
# ---------------------------------------------------------------------------

    rule generateHypnogram:
        input:
            eeg_file,
            motion_csv = expand("{output_dir}/Motion_Tracking/Motion_Outputs/{chan_ids}/{chan_ids}_motion.csv", 
                   output_dir = out_dir, chan_ids = mouse_ids),
        output:
            combined_hypnogram = out_dir + "/Hypnogram_Data/{curr_chan}/{curr_chan}_combined_hypnogram.csv",
        threads: 1
        resources:
            mem_qsub = "50G",
            job_time = "05:00:00",
        shell:
            "python3 {scripts_dir}/Hypnogram_Generator_Draft.py {in_dir} {out_dir} {batch_id} {wildcards.curr_chan}"

# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

# make a list of inputs for spike detection
seizure_non_seizure_inputs = list()

# if the user has indicated that they would also like to screen for seizures, add seizure data as an input file so that the spike detection that sorts by seizures file can be generated
if detect_spikes and detect_seizures == 'True':
    seizure_non_seizure_inputs.append(expand("{output_dir}/Event_Detection_Data/Seizures/{chan_ids}_Seizures.csv", 
                                               output_dir = out_dir, chan_ids = channel_ids))

# if the user opted to screen for spikes, include the corresponding rule; differentiates between threshold and rolling spikes as well
if detect_spikes == 'True':

    if detect_threshold_spikes == 'True':
        rule detectThresholdSpikes:
            input:
                seizure_non_seizure_inputs,
                eeg_file,
            output:
                threshold_spikes_output = out_dir + "/Event_Detection_Data/Spikes/Threshold_Spikes/{curr_chan}/{curr_chan}_threshold_spikes.csv", 
            threads: 1
            resources:
                mem_qsub = "50G",
                job_time = "05:00:00",
            shell:
                "python3 {scripts_dir}/Threshold_Spike_Detection.py {in_dir} {out_dir} {batch_id} {wildcards.curr_chan}"

    if detect_rolling_spikes == 'True':
        rule detectRollingSpikes:
            input:
                seizure_non_seizure_inputs,
                eeg_file,
            output:
                rolling_spikes_output = out_dir + "/Event_Detection_Data/Spikes/Rolling_Spikes/{curr_chan}/{curr_chan}_rolling_spikes.csv", 
            threads: 1
            resources:
                mem_qsub = "50G",
                job_time = "05:00:00",
            shell:
                "python3 {scripts_dir}/Rolling_Spike_Detection.py {in_dir} {out_dir} {batch_id} {wildcards.curr_chan}"

# if the user opted to screen for seizures, include the corresponding rule
if detect_seizures == 'True':
    rule detectSeizures:
        input:
            eeg_file,
        output:
            seizure_output = out_dir + "/Event_Detection_Data/Seizures/{curr_chan}_Seizures.csv", 
        threads: 1
        resources:
            mem_qsub = "4G",
            job_time = "05:00:00",
        shell:
            "python3 {scripts_dir}/Seizure_Detection.py {in_dir} {out_dir} {batch_id} {wildcards.curr_chan}"

# if the user opted to screen for swds, include the corresponding rule
if detect_swds == 'True':
    rule detectSWDs:
        input:
            eeg_file,
        output:
            swds_output = out_dir + "/Event_Detection_Data/SWDs/{curr_chan}_SWD_events.csv", 
        threads: 1
        resources:
            mem_qsub = "4G",
            job_time = "05:00:00",
        shell:
            "python3 {scripts_dir}/SWD_Detection.py {in_dir} {out_dir} {batch_id} {wildcards.curr_chan}"

# if the user opted to screen for spindles, include the corresponding rule
if detect_spindles == 'True':
    rule detectSpindles:
        input:
            eeg_file,
        output:
            spindles_output = out_dir + "/Event_Detection_Data/Spindles/{curr_chan}_Spindle_events.csv", 
        threads: 1
        resources:
            mem_qsub = "4G",
            job_time = "05:00:00",
        shell:
            "python3 {scripts_dir}/Spindle_Detection.py {in_dir} {out_dir} {batch_id} {wildcards.curr_chan}"

# if the user opted for any event detection, include the events detected as inputs in the FineMappedSpectral Analysis rule
# the inputs are necessary to specify so that the proper events are removed from the EEG data
if detect_spikes == 'True' or detect_seizures == 'True' or detect_spindles == 'True' or detect_swds == 'True':
    fine_mapped_inputs = list()
    if detect_spikes == 'True':
        if detect_threshold_spikes == 'True':
            fine_mapped_inputs.append(expand("{output_dir}/Event_Detection_Data/Spikes/Threshold_Spikes/{chan_ids}/{chan_ids}_threshold_spikes.csv", 
                                           output_dir = out_dir, chan_ids = channel_ids))
        if detect_rolling_spikes == 'True':
            fine_mapped_inputs.append(expand("{output_dir}/Event_Detection_Data/Spikes/Rolling_Spikes/{chan_ids}/{chan_ids}_rolling_spikes.csv", 
                                           output_dir = out_dir, chan_ids = channel_ids))
            
    if detect_swds == 'True':
        fine_mapped_inputs.append(expand("{output_dir}/Event_Detection_Data/SWDs/{chan_ids}_SWD_events.csv", 
                                           output_dir = out_dir, chan_ids = channel_ids))

    if detect_seizures == 'True':
        fine_mapped_inputs.append(expand("{output_dir}/Event_Detection_Data/Seizures/{chan_ids}_Seizures.csv", 
                                           output_dir = out_dir, chan_ids = channel_ids))

    if detect_spindles == 'True':
        fine_mapped_inputs.append(expand("{output_dir}/Event_Detection_Data/Spindles/{chan_ids}_Spindle_events.csv", 
                                           output_dir = out_dir, chan_ids = channel_ids))
# FineMappedSpectralAnalysis rule
    if video_exists != 'NO VIDEO':
        rule FineMappedSpectral:
            input:
                fine_mapped_inputs,
                expand("{output_dir}/Hypnogram_Data/{chan_ids}/{chan_ids}_combined_hypnogram.csv",
                                               output_dir = out_dir, chan_ids = channel_ids)
            output:
                fine_mapped = out_dir + "/Fine_Mapped_Spectral_Analysis/{curr_chan}/{curr_chan}_full_Fine_Mapped_Spectral_Analysis.png",
            threads: 1
            resources:
                mem_qsub = "4G",
                job_time = "05:00:00",
            shell:
                "python3 {scripts_dir}/Fine_Mapped_Spectral_Analysis.py {in_dir} {out_dir} {batch_id} {wildcards.curr_chan}"
            


# ---------------------------------------------------------------------------
# Filter events
# ---------------------------------------------------------------------------
# create a list of inputs for the filtered events rule; depends on rollings vs. threshold spikes
filtered_events_inputs = list()
if detect_rolling_spikes == 'True':
    filtered_events_inputs.append(expand("{output_dir}/Event_Detection_Data/Spikes/Rolling_Spikes/{chan_ids}/{chan_ids}_rolling_spikes.csv", 
                                           output_dir = out_dir, chan_ids = channel_ids))
if detect_threshold_spikes == 'True':
    filtered_events_inputs.append(expand("{output_dir}/Event_Detection_Data/Spikes/Threshold_Spikes/{chan_ids}/{chan_ids}_threshold_spikes.csv", 
                                          output_dir = out_dir, chan_ids = channel_ids))

# run findFilteredEvents rule if the user provided a video file
if video_exists != 'NO VIDEO':
    rule findFilteredEvents:
        input:
            combined_hypnogram = expand("{output_dir}/Hypnogram_Data/{chan_ids}/{chan_ids}_combined_hypnogram.csv",
                   output_dir = out_dir, chan_ids = channel_ids),
            filter_inputs = filtered_events_inputs,
        output:
            filtered_events = out_dir + "/Filtered_Epileptiform_Events/completion_messages/{curr_chan}_completion.csv",
        threads: 1
        resources:
            mem_qsub = "50G",
            job_time = "05:00:00",
        shell:
            "python3 {scripts_dir}/Filtered_Epileptiform_Events.py {in_dir} {out_dir} {batch_id} {wildcards.curr_chan}"  

# set outputs for final output csv file
FinalOutputInputs = list()           
if detect_spikes == 'True' or detect_seizures == 'True' or detect_spindles == 'True' or detect_swds == 'True':
    if detect_spikes == 'True':
        if detect_threshold_spikes == 'True':
            FinalOutputInputs.append(expand("{output_dir}/Event_Detection_Data/Spikes/Threshold_Spikes/{chan_ids}/{chan_ids}_threshold_spikes.csv", 
                                           output_dir = out_dir, chan_ids = channel_ids))
        if detect_rolling_spikes == 'True':
            FinalOutputInputs.append(expand("{output_dir}/Event_Detection_Data/Spikes/Rolling_Spikes/{chan_ids}/{chan_ids}_rolling_spikes.csv", 
                                           output_dir = out_dir, chan_ids = channel_ids))
            
    if detect_swds == 'True':
        FinalOutputInputs.append(expand("{output_dir}/Event_Detection_Data/SWDs/{chan_ids}_SWD_events.csv", 
                                           output_dir = out_dir, chan_ids = channel_ids))

    if detect_seizures == 'True':
        FinalOutputInputs.append(expand("{output_dir}/Event_Detection_Data/Seizures/{chan_ids}_Seizures.csv", 
                                           output_dir = out_dir, chan_ids = channel_ids))

    if detect_spindles == 'True':
        FinalOutputInputs.append(expand("{output_dir}/Event_Detection_Data/Spindles/{chan_ids}_Spindle_events.csv", 
                                           output_dir = out_dir, chan_ids = channel_ids))
if video_exists != 'NO VIDEO':
    FinalOutputInputs.append(expand("{output_dir}/Hypnogram_Data/{chan_ids}/{chan_ids}_combined_hypnogram.csv",
                                            output_dir = out_dir, chan_ids = channel_ids))
    FinalOutputInputs.append(expand("{output_dir}/Filtered_Epileptiform_Events/completion_messages/{chan_ids}_completion.csv",
                                               output_dir = out_dir, chan_ids = channel_ids))
# final output rule       
rule getFinalOutput:
        input:
            FinalOutputInputs,
        output:
            final_output = out_dir + "/Final_Output/" + batch_id + "_final_output.csv",
        threads: 1
        resources:
            mem_qsub = "50G",
            job_time = "05:00:00",
        shell:
            "python3 {scripts_dir}/Final_Statistics.py {in_dir} {out_dir} {batch_id}"  