import os
import mne
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load the EEG data from the .gdf file
ERPfile1 = 'StimuliVerificationTrials/data/BrianERP1.gdf'
ERPfile2 = 'StimuliVerificationTrials/data/BrianERP2.gdf'
ERPfile3 = 'StimuliVerificationTrials/data/BrianERP3.gdf'
ERPfile4 = 'StimuliVerificationTrials/data/BrianERP4.gdf'
ERPfile5 = 'StimuliVerificationTrials/data/BrianERP5.gdf'

# Define array of files
ERPfiles = [ERPfile5]

# Define EEG channels of interest
eeg_channels = ['FC1', 'FC2', 'CZ', 'C3', 'C4', 'CP1', 'CP2', 'PZ', 'P3', 'P4', 'FZ', 'FC1', 'FC2']
eeg_channels = ['FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'CZ', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2']
eeg_channels = ['FP1', 'FP2', 'F7', 'F8', 'F3', 'FZ', 'F4', 'FC1', 'FC2', 'C3', 'CZ', 'C4', 'CP5', 'CP1', 'P3', 'PZ']
# Define EOG channels
eog_channels = ['sens13', 'sens14', 'sens15']

# Define all channels of interest
channels_of_interests = ['FP1', 'FP2', 'F7', 'F8', 'F3', 'FZ', 'F4', 'FC1', 'FC2', 'C3', 'CZ', 'C4', 'CP5', 'CP1', 'P3', 'PZ', 'sens13']

# Events of interests
event_ids = [2, 3]

# Initialize an empty list to store ERP data for each channel
avg_all_base_channel_dict_arr = []
avg_all_target_channel_dict_arr = []

standard_montage = mne.channels.make_standard_montage('standard_1020')

for file in ERPfiles:
    # Extract only the file name from the full filepath for plot title
    file_name = os.path.basename(file)

    # read the raw data
    raw = mne.io.read_raw_gdf(file, preload=True, eog=eog_channels)
    # setting montage for naming
    raw.set_montage(standard_montage, match_case=False)

    # Before we look at each channel we need to pre-process the data and reject eye artifacts
    raw.pick_channels(channels_of_interests)

    # Apply the notch filter
    freq_to_notch = 60  # Replace with your desired notch frequency
    raw.notch_filter(freq_to_notch, picks='eeg')

    # Apply bandpass filter between 1 and 30 Hz
    raw.filter(l_freq=1, h_freq=10, picks='eeg')

    # Define events (assuming you have markers for the visual stimuli)
    events, _ = mne.events_from_annotations(raw)

    # Initialize an empty array to store ERP data each entry will be the average of all epochs for a certain channel
    erp_array_target = []
    erp_array_base = []

    # Initialize array of all epochs
    all_epochs = []

    # Segment the data into base and target epochs
    for eid in event_ids:
        # Epoch the data around the events
        if eid == 3:
            base_epochs = mne.Epochs(raw, events, eid, tmin=-0.25, tmax=0.5, baseline=(-0.25, -0.05), detrend=1, preload=True).copy()
            # throw out epochs who observe 200 or above on the EOG
            base_epochs.drop_bad(reject=dict(eog=200))
            all_epochs.insert(0, base_epochs)
        else:
            target_epochs = mne.Epochs(raw, events, eid, tmin=-0.25, tmax=0.5, baseline=(-0.25, -0.05), detrend=1, preload=True).copy()
            # throw out epochs who observe 200 or above on the EOG
            target_epochs.drop_bad(reject=dict(eog=200))
            all_epochs.insert(1, target_epochs)

    # Initialize an empty list to store ERP data for each channel
    avg_base_channel_dict = {}
    avg_target_channel_dict = {}

    for channel in eeg_channels:
        # plt.figure()

        for epoch_list in all_epochs:
            avg_channel_erp = epoch_list.average(picks=[channel])
            if epoch_list.events[0][2] == 3:
                avg_base_channel_dict[channel] = avg_channel_erp
                # plt.plot(avg_channel_erp.times, avg_channel_erp.data[0], label='Base Stimuli')
            else:
                avg_target_channel_dict[channel] = avg_channel_erp

    # Add in the average erp for a single channel
    avg_all_base_channel_dict_arr.append(avg_base_channel_dict)
    avg_all_target_channel_dict_arr.append(avg_target_channel_dict)

# Initialize dictionaries to store averaged ERPs for each channel across all files
grand_avg_base_channel_dict = {channel: [] for channel in eeg_channels}
grand_avg_target_channel_dict = {channel: [] for channel in eeg_channels}


# Loop through each file
for avg_base_channel_dict, avg_target_channel_dict in zip(avg_all_base_channel_dict_arr, avg_all_target_channel_dict_arr):
    # Loop through each channel
    for channel in eeg_channels:
        grand_avg_base_channel_dict[channel].append(avg_base_channel_dict[channel].data[0])
        grand_avg_target_channel_dict[channel].append(avg_target_channel_dict[channel].data[0])

# Compute grand averages across all files for each channel
grand_avg_base_channel_dict = {channel: sum(erps) / len(erps) for channel, erps in grand_avg_base_channel_dict.items()}
grand_avg_target_channel_dict = {channel: sum(erps) / len(erps) for channel, erps in grand_avg_target_channel_dict.items()}

# Determine the number of rows and columns for the subplot matrix
n_channels = len(eeg_channels)
n_cols = 4
n_rows = (n_channels + n_cols - 1) // n_cols

# Create a matrix of subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))  # Adjust figsize as needed

# Flatten the 2D array of subplots
axes = axes.flatten()

# Plot grand averages in the subplots
for i, channel in enumerate(eeg_channels):
    ax = axes[i]
    ax.plot(
        avg_base_channel_dict[channel].times * 1000,
        grand_avg_base_channel_dict[channel],
        label='Grand Average Base Stimuli'
    )
    ax.plot(
        avg_target_channel_dict[channel].times * 1000,
        grand_avg_target_channel_dict[channel],
        label='Grand Average Target Stimuli'
    )
    ax.axvline(x=250, color='red', linestyle='--', label='Vertical Line at 250ms')  # Add a vertical line at 250ms
    ax.axvline(x=300, color='green', linestyle='--', label='Vertical Line at 300ms')  # Add a vertical line at 300ms
    ax.axvline(x=350, color='red', linestyle='--', label='Vertical Line at 350ms')  # Add a vertical line at 350ms
    # ax.invert_yaxis()  # Invert the y-axis (positive down)
    ax.set_title(f'Grand Average ERP for {channel}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (uV)')
    # ax.yaxis.set_major_locator(plt.MultipleLocator(1))  # Set y-axis ticks every 1uV
    ax.xaxis.set_major_locator(plt.MultipleLocator(100))  # Set x-axis ticks every 100ms

# Adjust layout
plt.tight_layout()

# Add a common legend outside the subplots
fig.subplots_adjust(top=0.94)  # Adjust top margin
fig.legend(['Brian Grand Average Base Stimuli', 'Brian Grand Average Target Stimuli'], loc='right', bbox_to_anchor=(0.5, 0.98), fancybox=True, shadow=True, ncol=2)

# Show the figure
plt.show()