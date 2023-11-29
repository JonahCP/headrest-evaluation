import os
import numpy as np
import mne
import matplotlib.pyplot as plt

# Will be cleaning up this file, still a work in progress

# Below Plots an average ERP waveform for each file, utilizing your channels of interest.
# Shows for both the target - event ID 1 and base - event ID 2
# Final Graph shows the final averaged ERP waveform of all graphs

ERPfile = 'headrest-evaluation/analysis/StimuliVerificationTrials/11_17_23_364DEEGtests/JasonTestErp.gdf'
ERPfile1 = 'headrest-evaluation/analysis/StimuliVerificationTrials/11_17_23_364DEEGtests/JasonErp1.gdf'
ERPfile2 = 'headrest-evaluation/analysis/StimuliVerificationTrials/11_17_23_364DEEGtests/JasonErp2.gdf'
ERPfile3 = 'headrest-evaluation/analysis/StimuliVerificationTrials/11_17_23_364DEEGtests/JasonErp3.gdf'
ERPfile4 = 'headrest-evaluation/analysis/StimuliVerificationTrials/11_17_23_364DEEGtests/JasonErp4.gdf'

# Define EOG channels
eog_channels = ['sens13', 'sens14', 'sens15']

# Load and process each file
# Define the EEG channels you are interested in
channels_of_interest = ['FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2', 'M1', 'M2']
# channels_of_interest = ['F7', 'FPZ', 'F8', 'F3', 'FZ', 'F4', 'C3', 'CZ', 'C4', 'P3', 'PZ', 'P4']

standard_montage = mne.channels.make_standard_montage('standard_1020')

# Initialize lists to store average responses and labels
all_average_responses = []
all_labels = []

for event_id_of_interest in [1, 2]:
    # Initialize list to store individual waveforms
    individual_waveforms = []

    for file in [ERPfile, ERPfile1, ERPfile2, ERPfile3, ERPfile4]:
        raw = mne.io.read_raw_gdf(file, preload=True, eog=eog_channels)
        raw.set_montage(standard_montage, match_case=False)

        # Apply the notch filter
        freq_to_notch = 60  # Replace with your desired notch frequency
        raw.notch_filter(freq_to_notch, picks='eeg')

        # Apply the bandpass filter
        raw.filter(l_freq=.1, h_freq=30)

        # Epoch the data
        events, event_id = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0.1), detrend=1, preload=True, event_repeated='drop')

        # Average across all EEG channels
        average_response = epochs[event_id_of_interest].average(picks='eeg')

        # Plot the average response for the current file
        times = average_response.times * 1000  # Convert to milliseconds
        data = average_response.data.mean(axis=0)  # Take the mean across all channels
        plt.plot(times, data, label=f'{file.split("/")[-1]} - Event ID {event_id_of_interest}')

        # Store individual waveform for later use
        individual_waveforms.append(data)

        # Store average response and label for later use
        all_average_responses.append(data)
        all_labels.append(f'{file.split("/")[-1]} - Event ID {event_id_of_interest}')

    # Customize the plot for each event_id_of_interest
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uV)')
    plt.title(f'Average Response Across All Channels for Each File (Event ID {event_id_of_interest})')
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(0, max(times) + 100, 100))  # Add tick marks every 100ms
    plt.show()

    # Create a final graph that averages the individual waveforms for each event_id_of_interest
    plt.figure(figsize=(12, 6))
    plt.plot(times, np.array(individual_waveforms).mean(axis=0), label=f'Average Across Files - Event ID {event_id_of_interest}')

    # Customize the final plot
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uV)')
    plt.title(f'Average Response Across All Channels for All Files (Event ID {event_id_of_interest})')
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(0, max(times) + 100, 100))  # Add tick marks every 100ms
    plt.show()

# Create a final graph that averages the individual waveforms for all event_id_of_interest
plt.figure(figsize=(12, 6))
plt.plot(times, np.array(all_average_responses).mean(axis=0), label='Average Across Files - All Event IDs')

# Customize the final plot
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (uV)')
plt.title('Average Response Across All Channels for All Files (All Event IDs)')
plt.legend()
plt.grid(True)
plt.xticks(np.arange(0, max(times) + 100, 100))  # Add tick marks every 100ms
plt.show()

##------------------------------------------------------------------------------------------------------------------##

# Shows all selected channels averaged across all files, the one final averaged ERP
# i.e you will see a chart for every single individual channel which is a result of averaging all data collected
# from those nodes in every file/trial

# Load and process each file
channels_of_interest = ['FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2', 'M1', 'M2']

# Testing some channels of interest
# channels_of_interest = ['F3', 'FZ', 'F4', 'FC1', 'FC2', 'C3', 'CZ', 'C4', 'CP1', 'CP2', 'P3', 'PZ', 'P4', 'POZ', 'O1', 'O2']
# channels_of_interest = ['FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2']
# channels_of_interest = ['F7', 'FPZ', 'F8', 'F3', 'FZ', 'F4', 'C3', 'CZ', 'C4', 'P3', 'PZ', 'P4']

standard_montage = mne.channels.make_standard_montage('standard_1020')

# Set up the subplot grid
n_channels = len(channels_of_interest)
n_cols = 4  # Number of columns in the grid
n_rows = int(np.ceil(n_channels / n_cols))  # Number of rows needed
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.5 * n_rows))

for idx, channel in enumerate(channels_of_interest):
    row = idx // n_cols
    col = idx % n_cols

    axes[row, col].set_title(channel, fontsize=12)  # Adjust title font size
    axes[row, col].set_xlabel('Time (s)')
    axes[row, col].set_ylabel('Amplitude (uV)')

    all_epochs = []
    for file in [ERPfile1, ERPfile2, ERPfile3, ERPfile4]:
        raw = mne.io.read_raw_gdf(file, preload=True, eog=eog_channels)
        raw.set_montage(standard_montage, match_case=False)

        raw.pick_channels([channel])

        # Apply the notch filter
        freq_to_notch = 60  # Replace with your desired notch frequency
        raw.notch_filter(freq_to_notch, picks='eeg')

        # Apply the bandpass filter
        raw.filter(l_freq=.1, h_freq=30)

        # Epoch the data
        events, event_id = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0.1), detrend=1, preload=True,
                            event_repeated='drop')

        all_epochs.append(epochs)

    # Concatenate the epochs across files
    concatenated_epochs = mne.concatenate_epochs(all_epochs)

    # Specify the event ID of interest
    event_id_of_interest = 1

    # Average across all EEG channels
    average_response = concatenated_epochs[event_id_of_interest].average(picks='eeg')

    # Plot the average response for the current channel in its subplot
    times = average_response.times
    data = average_response.data.squeeze()
    axes[row, col].plot(times, data, label=channel)

    # Set more tick markers on the x-axis
    axes[row, col].set_xticks(np.arange(0, max(times) + 0.1, 0.1))

# Adjust layout
plt.tight_layout(pad=4)  # Increase the padding
plt.show()

# Plot individual lines for each channel
plt.figure(figsize=(12, 6))
for channel in channels_of_interest:
    picked_epochs = []

    for epochs in all_epochs:
        # Check if the channel is present before picking
        if channel in epochs.ch_names:
            picked_epochs.append(epochs.pick_channels([channel]))

    # Check if any epochs were picked for the current channel
    if picked_epochs:
        # Concatenate the epochs across files for the current channel
        concatenated_epochs_channel = mne.concatenate_epochs(picked_epochs)

        # Average across all files for the current channel
        average_response_channel = concatenated_epochs_channel.average()

        # Plot the average response for the current channel
        times = average_response_channel.times
        data = average_response_channel.data.squeeze()
        plt.plot(times, data, label=channel)

# Plot the average response for all EEG channels as a single line
average_response_all_channels = concatenated_epochs.average()
times_all_channels = average_response_all_channels.times
data_all_channels = average_response_all_channels.data.squeeze()
plt.plot(times_all_channels, data_all_channels, label='Average All Channels', linewidth=2)

# Customize the plot
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.title('Average Response Across All Channels for Target Stimuli')
plt.legend()
plt.grid(True)
plt.show()