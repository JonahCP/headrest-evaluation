import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

# Replace 'your_file.gdf' with the actual filename
ERPfile = 'headrest-evaluation/analysis/StimuliVerificationTrials/11_17_23_364DEEGtests/JasonTestErp.gdf'
ERPfile1 = 'headrest-evaluation/analysis/StimuliVerificationTrials/11_17_23_364DEEGtests/JasonErp1.gdf'
ERPfile2 = 'headrest-evaluation/analysis/StimuliVerificationTrials/11_17_23_364DEEGtests/JasonErp2.gdf'
ERPfile3 = 'headrest-evaluation/analysis/StimuliVerificationTrials/11_17_23_364DEEGtests/JasonErp3.gdf'
ERPfile4 = 'headrest-evaluation/analysis/StimuliVerificationTrials/11_17_23_364DEEGtests/JasonErp4.gdf'

# Define EOG channels
eog_channels = ['sens13', 'sens14', 'sens15']

# Load and process each file
# channels_of_interest = ['FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2', 'M1', 'M2']
# channels_of_interest = ['AF3', 'AF4', 'F3', 'F1', 'FZ', 'F2', 'F4', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C3', 'C1', 'CZ', 'C2', 'C4', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P3', 'P1', 'PZ', 'P2', 'P4', 'PO3', 'POZ', 'PO4', 'O1', 'O2']
# channels_of_interest = ['F3', 'FZ', 'F4', 'FC1', 'FC2', 'C3', 'CZ', 'C4', 'CP1', 'CP2', 'P3', 'PZ', 'P4', 'POZ', 'O1', 'O2']
# channels_of_interest = ['FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2']
channels_of_interest = ['F7', 'FPZ', 'F8', 'F3', 'FZ', 'F4', 'C3', 'CZ', 'C4', 'P3', 'PZ', 'P4']

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
        epochs = mne.Epochs(raw, events, event_id, tmin=-.2, tmax=.6, baseline=(-.2, 0.1), detrend=1, preload=True, event_repeated='drop')

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
plt.title('Average Response Across All Channels')
plt.legend()
plt.grid(True)
plt.show()


#
# # Replace 'your_file.gdf' with the actual filename
# ERPfile1 = 'headrest-evaluation/analysis/StimuliVerificationTrials/11_17_23_364DEEGtests/JasonErp1.gdf'
# ERPfile2 = 'headrest-evaluation/analysis/StimuliVerificationTrials/11_17_23_364DEEGtests/JasonErp2.gdf'
# ERPfile3 = 'headrest-evaluation/analysis/StimuliVerificationTrials/11_17_23_364DEEGtests/JasonErp3.gdf'
# ERPfile4 = 'headrest-evaluation/analysis/StimuliVerificationTrials/11_17_23_364DEEGtests/JasonErp4.gdf'
#
# # Define EOG channels
# eog_channels = ['sens13', 'sens14', 'sens15']
#
# # Define the EEG channels you are interested in
# channels_of_interest = ['PZ', 'P3', 'P4', 'CZ', 'CP5', 'OZ', 'P8', 'POZ', 'O1', 'OZ', 'O2']
#
# # Load the data
# raw = mne.io.read_raw_gdf(ERPfile3, preload=True, eog=eog_channels)
#
# # raw.pick_channels(channels_of_interest)
#
# # Apply the notch filter
# freq_to_notch = 60  # Replace with your desired notch frequency
# raw.notch_filter(freq_to_notch, picks='eeg')
#
# # Filter the data
# raw.filter(l_freq=.1, h_freq=30)
#
# # Epoch the data (replace events and event_id with your actual event information)
# events, event_id = mne.events_from_annotations(raw)
#
# # print(raw.annotations)
# # print(events)
# # print(event_id)
#
# # Specify tmin and tmax in milliseconds
# epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=.5, baseline=(0, 0.1), preload=True, event_repeated='drop')
#
# print(epochs)
#
# # Access the events array
# events_array = epochs.events
#
# # Print the contents of each event
# for event in events_array:
#     print(f"Sample Index: {event[0]}, Event Value: {event[1]}, Event ID: {event[2]}")
#
# # Specify the event ID of interest
# event_id_of_interest = 1
#
# # Plot the average response for epochs with the specified event ID
# epochs[event_id_of_interest].average().plot()
#
# # Plot the EOG channel data
# raw.plot(duration=10, n_channels=len(eog_channels))  # Adjust scalings as needed
#
# # Show the plot
# mne.viz.tight_layout()
# mne.viz.show()