import os
import mne
import pandas as pd
from pandas import DataFrame, read_csv
import numpy
import re
import numpy as np
from matplotlib import pyplot as plt
from autoreject import get_rejection_threshold


def format_events_frame(EventsDataFrame):
    # Drop the first column since it's the same as the index
    EventsDataFrame = EventsDataFrame.drop(EventsDataFrame.columns[0], axis=1)

    # Convert 'datetimes' column to datetime format
    EventsDataFrame['datetimes'] = pd.to_datetime(EventsDataFrame['datetimes'], format='%Y-%m-%d %H.%M.%S.%f',
                                                  errors='coerce', utc=True)

    # Gets the seconds that have passed since the trial started
    EventsDataFrame['seconds_since_start'] = (
            EventsDataFrame['datetimes'] - EventsDataFrame['datetimes'].min()).dt.total_seconds()

    # Create 'TID' column based on conditions
    conditions = [
        (EventsDataFrame['events'] == 'START') | (EventsDataFrame['events'] == 'END'),
        (EventsDataFrame['events'] == 'STANDARD'),
        (EventsDataFrame['events'] == 'TARGET'),
        (EventsDataFrame['events'] == 'KEY PRESS')
    ]

    values = [1, 10, 20, 30]

    # Adds the TID next to the events
    EventsDataFrame['TID'] = np.select(conditions, values, default=np.nan)

    # Reorder columns
    EventsDataFrame = EventsDataFrame[
        ['datetimes', 'seconds_since_start', 'events', 'TID'] + [col for col in EventsDataFrame.columns if
                                                                 col not in ['datetimes', 'seconds_since_start',
                                                                             'events', 'TID']]]

    return EventsDataFrame


def format_raw_data_frame(DataFrame, startTime):
    # Convert 'datetime' column to datetime format
    DataFrame['datetime'] = pd.to_datetime(DataFrame['datetime'], format='%Y-%m-%d %H.%M.%S.%f', errors='coerce',
                                           utc=True)

    # Gets the seconds that have passed since the trial started
    DataFrame['seconds_since_start'] = (
            DataFrame['datetime'] - startTime).dt.total_seconds()

    # Get raw channel data
    raw_channel_data = DataFrame[['chnl-1-raw', 'chnl-2-raw', 'chnl-3-raw', 'datetime', 'seconds_since_start']].values.T

    return raw_channel_data


def get_trimmed_files():
    base_directory = 'hr_data'
    # The line below searches the hr_data directory for other directories (which contain the test data)
    directories = [f for f in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, f))]

    files = []

    for directory in directories:
        for i in range(1, 9):
            file_path = f'{base_directory}/{directory}/erp{i}_trim.csv'
            files.append(file_path)

    return files


def get_timestamps():
    base_directory = 'hr_data'
    # The line below searches the hr_data directory for other directories (which contain the test data)
    directories = [f for f in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, f))]

    files = []

    for directory in directories:
        subject_base_directory = f'{base_directory}/{directory}'
        # Does the same as before except only grabs the erp directories for a single subject
        erp_directories = [f for f in os.listdir(subject_base_directory)
                           if os.path.isdir(os.path.join(subject_base_directory, f)) and 'erp' in f]

        for erp_directory in erp_directories:
            erp_directory_path = f'{base_directory}/{directory}/{erp_directory}'
            erp_timestamp = [f for f in os.listdir(erp_directory_path) if re.match(r'erp_timestamps_.*\.csv', f)][0]

            files.append(f'{base_directory}/{directory}/{erp_directory}/{erp_timestamp}')

    return files


TrimmedERPFiles = get_trimmed_files()
ERPTimestamps = get_timestamps()

FileMatrix = [TrimmedERPFiles, ERPTimestamps]

FileDataFrame = pd.DataFrame(FileMatrix)

# Initialize an empty list to store ERP data for each channel
list_of_avg_base_stimuli_across_all_trials = []
list_of_avg_target_stimuli_across_all_trials = []

num_of_targets = 0
num_of_base = 0

for i in range(FileDataFrame.columns.size):
    ERPDataFrame = read_csv(FileDataFrame[i].iloc[0])
    ERPEventsDataFrame = read_csv(FileDataFrame[i].iloc[1])

    events_info = format_events_frame(ERPEventsDataFrame)

    # start time of the trial
    start_time = events_info['datetimes'].iloc[0]

    # Pass in the ERPDataFrame and the start time of the trial to have an accurate start time for both
    raw_array = format_raw_data_frame(ERPDataFrame, start_time)

    ch_names = ['Channel 1', 'Channel 2', 'Channel 3']

    ch_types = ['eeg', 'eeg', 'eeg']

    sampling_frequency = 206

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sampling_frequency)

    raw = mne.io.RawArray(raw_array[:3, :], info).set_meas_date(start_time)

    onset_array = events_info['seconds_since_start'].values

    description_array = events_info['TID'].values.astype(int)

    annotations = mne.Annotations(onset=onset_array, description=description_array, duration=0.001953125,
                                  orig_time=start_time)

    raw.set_annotations(annotations)

    # Begin analysis
    raw.pick_channels(ch_names)

    # raw.compute_psd().plot()

    raw.filter(5, 30, fir_design='firwin', fir_window='hamming')

    # raw.compute_psd().plot()

    # Creating filter
    fir_coefs = mne.filter.create_filter(
        data=raw.get_data(),  # data is only used for sanity checking, not strictly needed
        sfreq=206,  # sfreq of your data in Hz
        l_freq=5,
        h_freq=30,  # assuming a lowpass of 40 Hz
        # filter_length=10001,
        method='fir',
        fir_window='hamming',
        fir_design='firwin',
        verbose=True)

    # See the printed log for the transition bandwidth and filter length.
    # Alternatively, get the filter length through:
    filter_length = fir_coefs.shape[0]

    # mne.viz.plot_filter(fir_coefs, 206)

    # Define events based on your experimental paradigm
    events, event_id = mne.events_from_annotations(raw)

    # Where 2 is base and 3 is target stimuli
    event_ids_of_interest = [2, 3]

    # Picks are just defining what type of channels we are using, in this case EEG and EOG
    picks = mne.pick_types(raw.info, eeg=True)

    # Create epochs around events of interest (e.g., visual stimuli)
    base_epochs = None
    target_epochs = None

    # Define rejection thresholds for each channel
    # rejection_thresholds = {'Channel 1': 5000, 'Channel 2': 8000, 'Channel 3': 8000}

    for eid in event_ids_of_interest:
        final_epochs = mne.Epochs(raw, events, eid, tmin=-0.25, tmax=0.75,
                                  preload=True, event_repeated='merge')

        reject = get_rejection_threshold(final_epochs)['eeg']

        final_epochs.drop_bad(reject={'eeg': reject})

        # epochs_specific_channel = final_epochs.pick_channels(['Channel 1'])
        # epochs_specific_channel.plot()

        # final_epochs.plot(picks=picks, events=events, scalings=scalings)
        if final_epochs.__len__() == 0:
            break

        final_epochs = final_epochs.apply_baseline(baseline=(-.25, 0))

        if eid == 2:
            base_epochs = final_epochs.copy()
            num_of_base += base_epochs.events.shape[0]
        else:
            target_epochs = final_epochs.copy()
            num_of_targets += target_epochs.events.shape[0]

    # base_epochs.plot(picks=picks, events=events, scalings='auto')
    # target_epochs.plot(picks=picks, events=events, scalings='auto')

    # The following are evoked objects in mne
    if base_epochs is not None:
        base_epochs_avg = base_epochs.average(picks='eeg')
        list_of_avg_base_stimuli_across_all_trials.append(base_epochs_avg)
    if target_epochs is not None:
        target_epochs_avg = target_epochs.average(picks='eeg')
        list_of_avg_target_stimuli_across_all_trials.append(target_epochs_avg)

# base_epochs_avg.plot(picks='eeg')
# target_epochs_avg.plot(picks='eeg')

# Take the grand average of the base and target stimulis
grand_average_base_stimuli = mne.grand_average(list_of_avg_base_stimuli_across_all_trials)
grand_average_target_stimuli = mne.grand_average(list_of_avg_target_stimuli_across_all_trials)

# Calculate the number of rows and columns for the grid
num_rows = 3
num_cols = 1

# Calculate the number of subplots needed
num_subplots = int(np.ceil(len(ch_names) / (num_rows * num_cols)))

# Determine y-axis limits across all channels
min_y = float('inf')
max_y = float('-inf')

for ch_name in ch_names:
    base_data = grand_average_base_stimuli.get_data(picks=ch_name)
    target_data = grand_average_target_stimuli.get_data(picks=ch_name)

    min_y = min(min_y, np.min([base_data, target_data]))
    max_y = max(max_y, np.max([base_data, target_data]))

# Loop over subplots
for subplot_idx in range(num_subplots):
    start_channel_idx = subplot_idx * num_rows * num_cols
    end_channel_idx = (subplot_idx + 1) * num_rows * num_cols

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 12))

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    # Loop over channels and plot on separate subplots
    for idx, ch_name in enumerate(ch_names[start_channel_idx:end_channel_idx]):
        base_data = grand_average_base_stimuli.get_data(picks=ch_name)
        target_data = grand_average_target_stimuli.get_data(picks=ch_name)

        # Plot the data for the current channel on the corresponding subplot
        axes[idx].plot(grand_average_base_stimuli.times * 1000, base_data[0], label='Base Stimuli', color='blue')
        axes[idx].plot(grand_average_target_stimuli.times * 1000, target_data[0], label='Target Stimuli', color='red')

        # Add vertical lines at specific time points
        axes[idx].axvline(x=0, color='blue', linestyle='--', label='Stimuli Shown')
        axes[idx].axvline(x=300, color='green', linestyle='--', label='Vertical Line at 300ms')
        # axes[idx].axvline(x=430, color='green', linestyle='--', label='Vertical Line at 430ms')

        # Add a horizontal line at y=0
        axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=1, label='Zero Line')

        # Set labels and title
        axes[idx].set_title(f'Grand Averaged Epochs for {ch_name}')
        axes[idx].set_xlabel('Time (ms)')
        axes[idx].set_ylabel('Amplitude (uV)')

        # Invert the y-axis to flip the negative values upwards
        # axes[idx].invert_yaxis()

        # Set y-axis limits
        # axes[idx].set_ylim(min_y, max_y)

        # Set x-axis limits to 0 to 500 milliseconds
        # axes[idx].set_xlim(0, 500)

        # Add legend
        axes[idx].legend()

    # Adjust layout with auto spacing
    plt.tight_layout()

    # Show the plot
    plt.show()
