import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load the EEG data from the .gdf file
ERPfile1 = 'StimuliVerificationTrials/data/BrianERP1.gdf'
ERPfile2 = 'StimuliVerificationTrials/data/BrianERP2.gdf'
ERPfile3 = 'StimuliVerificationTrials/data/BrianERP3.gdf'
ERPfile4 = 'StimuliVerificationTrials/data/BrianERP4.gdf'
ERPfile5 = 'StimuliVerificationTrials/data/BrianERP5.gdf'

# Define array of files
ERPfiles = [ERPfile4, ERPfile5]

# Define EEG channels of interest
eeg_channels = ['FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'CZ', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2']
# Define EOG channels
eog_channels = ['sens13', 'sens14', 'sens15']

# Create an empty list to store epochs
all_epochs = []

# Iterate through files
for ERPfile in ERPfiles:
    # Read the raw data
    raw = mne.io.read_raw_gdf(ERPfile, preload=True, eog=eog_channels)

    # Apply the notch filter
    freq_to_notch = 60  # Replace with your desired notch frequency
    raw.notch_filter(freq_to_notch, picks='eeg')

    # Apply bandpass filter between 1 and 30 Hz
    raw.filter(l_freq=1, h_freq=30, picks='eeg')

    # Define events (assuming you have markers for the visual stimuli)
    events, _ = mne.events_from_annotations(raw)

    event_id_target = 2
    event_id_base = 3

    # Epoch the data around the events for target stimuli (event_id = 2)
    epochs_target = mne.Epochs(raw, events, event_id_target, tmin=-0.25, tmax=0.8, baseline=(-0.25, -0.05), detrend=1,
                               preload=True)

    # Highlight the start and end of each epoch using annotations for target stimuli
    for onset in epochs_target.events[:, 0]:
        duration = 1.5  # Set the duration of the epoch
        raw.annotations.append(onset=onset / raw.info['sfreq'], duration=duration, description='Target Epoch')
        duration = 0.5
        raw.annotations.append(onset=onset / raw.info['sfreq'], duration=duration, description='Stimuli Duration')

    # Epoch the data around the events for baseline stimuli (event_id = 3)
    epochs_base = mne.Epochs(raw, events, event_id_base, tmin=-0.25, tmax=0.8, baseline=(-0.25, -0.05), detrend=1,
                             preload=True)

    # Highlight the start and end of each epoch using annotations for baseline stimuli
    for onset in epochs_base.events[:, 0]:
        duration = 1.5  # Set the duration of the epoch
        raw.annotations.append(onset=onset / raw.info['sfreq'], duration=duration, description='Baseline Epoch')
        duration = 0.5
        raw.annotations.append(onset=onset / raw.info['sfreq'], duration=duration, description='Stimuli Duration')

    raw.plot(events=events, scalings='auto')

    # Create epochs around events
    epochs = mne.Epochs(raw, events, event_id=None, tmin=0, tmax=.5, baseline=None, detrend=1, preload=True, event_repeated='drop')

    epochs.pick_channels(['sens13'])

    # Iterate through epochs and check for EOG peaks
    for epoch in epochs.iter_evoked(copy=True):

        epoch_data = epoch.data[0]

        # Set the threshold for EOG peak detection (adjust as needed)
        eog_threshold = -21200  # You can adjust this threshold based on your data

        # Use a more robust peak detection algorithm
        peak_indices, _ = find_peaks(epoch_data, height=eog_threshold)

        for peak_index in peak_indices:
            # Find the peak time within the specified time window around the event
            onset_time = peak_index / raw.info['sfreq']

            # Annotate the epoch with EOG spike at the peak time
            # raw.annotations.append(onset=onset_time, duration=.01, description='EOG Spike')

        # Append the epochs from the current file to the list
        all_epochs.append(epoch)
        all_epochs.append(epoch.copy())

    if all_epochs[0] == all_epochs[1]:
        raw.plot(events=events)
    raw.plot(events=events)

# Plot the first file's data with annotations
# all_epochs[0].plot(events=True, event_color='cyan', scalings='auto', show_options=True, butterfly=False, color='black')
plt.show()



event_id_target = 2
event_id_base = 3


# Epoch the data around the events for target stimuli (event_id = 2)
epochs_target = mne.Epochs(raw, events, event_id_target, tmin=-0.25, tmax=0.8, baseline=(-0.25, -0.05), detrend=1, preload=True)

# Highlight the start and end of each epoch using annotations for target stimuli
for onset in epochs_target.events[:, 0]:
    duration = 1.5  # Set the duration of the epoch
    raw.annotations.append(onset=onset / raw.info['sfreq'], duration=duration, description='Target Epoch')
    duration = 0.5
    raw.annotations.append(onset=onset / raw.info['sfreq'], duration=duration, description='Stimuli Duration')

# Epoch the data around the events for baseline stimuli (event_id = 3)
epochs_base = mne.Epochs(raw, events, event_id_base, tmin=-0.25, tmax=0.8, baseline=(-0.25, -0.05), detrend=1, preload=True)

# Highlight the start and end of each epoch using annotations for baseline stimuli
for onset in epochs_base.events[:, 0]:
    duration = 1.5  # Set the duration of the epoch
    raw.annotations.append(onset=onset/raw.info['sfreq'], duration=duration, description='Baseline Epoch')
    duration = 0.5
    raw.annotations.append(onset=onset / raw.info['sfreq'], duration=duration, description='Stimuli Duration')

# Define the description of the epoch in our case 'Stimuli Duration'
description_of_interest = 'Stimuli Duration'

# Epoch the data around all stimuli
all_epochs = mne.Epochs(raw, events=None, event_id=None, tmin=0, tmax=.5, detrend=1, preload=True)

print(raw.annotations)


# # Epoch the data around the events for baseline stimuli (event_id = 3)
# all_epochs = mne.Epochs(raw, events, [event_id_base, event_id_target], tmin=-0.25, tmax=0.8, baseline=(-0.25, -0.05), detrend=1, preload=True)
#
# # Define the time window for EOG peak check (in seconds)
# eog_check_window = 0.5
#
# # Create a list to store epoch data for EOG channels
# all_epoch_data = []
#
# # Highlight the start and end of each bad epoch using annotations for eye blinks
# for epoch_index in range(len(all_epochs)):
#
#     # Get epoch data for eog channels for peak detection
#     single_epoch_data = all_epochs[epoch_index:epoch_index + 1].get_data(picks=eog_channels)
#
#     # Append the data to the list
#     all_epoch_data.append(single_epoch_data)



# Calculate the maximum EOG amplitude within the time window
    # max_eog_amplitude = np.max(np.abs(epoch_data))
    #
    # # Check if the maximum amplitude exceeds the threshold
    # if max_eog_amplitude > eog_threshold:
    #     # Annotate the epoch to be dropped
    #     raw.annotations.append(onset=0, duration=eog_check_window, description='bad Epoch (Massive EOG Spike)')
    #
    #     # Add the epoch to the list
    #     epochs_to_drop.append(epoch)


# Now, you can plot the raw data with the combined annotations
# raw.plot(events=events, event_color='cyan', scalings='auto', show_options=True, block=True, butterfly=False, color='black')
# plt.show()


# # Convert onset to seconds
#     onset_sec = onset / raw.info['sfreq']
#
#     # Define the time window for EOG peak check (in seconds)
#     eog_check_window = 0.5
#
#     # Crop the epoch data for the specified time window
#     epoch_data = epochs_target.copy().crop(tmin=onset_sec, tmax=onset_sec + eog_check_window).get_data(picks=eog_channels)
#
#     # Calculate the maximum EOG amplitude within the time window
#     max_eog_amplitude = np.max(np.abs(epoch_data))
#
#     # Check if the maximum amplitude exceeds the threshold
#     if max_eog_amplitude > eog_threshold:
#         # Annotate the epoch to be dropped
#         raw.annotations.append(onset=onset_sec, duration=eog_check_window, description='bad Epoch (Massive EOG Spike)')
#
#         # Add the epoch to the list
#         epochs_to_drop.append(onset)


#
# # Check if the maximum amplitude exceeds the threshold
# if max_eog_amplitude > eog_threshold:
#     # Annotate the epoch to be dropped
#     raw.annotations.append(onset=onset_sec, duration=eog_check_window, description='bad Epoch (Massive EOG Spike)')
#
#     # Add the epoch to the list
#     epochs_to_drop.append(onset)
#
#
#



















#
# # Events of interests
# event_ids = [2, 3]
#
# # Initialize an empty array to store ERP data each entry will be the average of all epochs for a certain channel
# erp_array_target = []
# erp_array_base = []
#
# standard_montage = mne.channels.make_standard_montage('standard_1020')
#
# for file in ERPfiles:
#     # Extract only the file name from the full filepath for plot title
#     file_name = os.path.basename(file)
#
#     # read the raw data
#     raw = mne.io.read_raw_gdf(file, preload=True, eog=eog_channels)
#     # setting montage for naming
#     raw.set_montage(standard_montage, match_case=False)
#
#     # raw.set_eeg_reference(ref_channels=['CZ'])
#
#     for channel in eeg_channels:
#
#         raw_copy = raw.copy()
#         # picking the individual channel, it is in square brackets since its required to be in a list
#         raw_copy.pick_channels([channel])
#
#         # Apply the notch filter
#         freq_to_notch = 60  # Replace with your desired notch frequency
#         raw_copy.notch_filter(freq_to_notch, picks='eeg')
#
#         # Apply bandpass filter between 1 and 30 Hz
#         raw_copy.filter(l_freq=1, h_freq=30, picks='eeg')
#
#         # Define events (assuming you have markers for the visual stimuli)
#         events, _ = mne.events_from_annotations(raw_copy)
#
#         for eid in event_ids:
#
#             # Epoch the data around the events
#             epochs = mne.Epochs(raw_copy, events, eid, tmin=-0.25, tmax=0.7, baseline=(-0.25, -0.05), detrend=1, preload=True)
#
#             # Plot the average ERP before artifact rejection
#             erp = epochs.average()
#
#             if eid == 2:
#                 # Adds the averaged ERP waveform to the array of targets from each
#                 erp_array_target.append(erp)
#                 # Plot the erp waveform,specifying the y-axis lim and giving it a title
#                 erp.plot(titles=channel + " - Target Stimuli: " + file_name, xlim='tight', ylim=dict(eeg=(-2e7, 2e7)), time_unit='ms', spatial_colors='auto')
#             else:
#                 # Adds the averaged ERP waveform to the array of targets from each
#                 erp_array_base.append(erp)
#                 # Plot the erp waveform,specifying the y-axis lim and giving it a title
#                 erp.plot(titles=channel + " - Base Stimuli: " + file_name, ylim=dict(eeg=(-2e7, 2e7)), time_unit='ms')  # Set y-axis scale to 1e-6
#
#             plt.show()
#
#
#
#
# # Compute and plot the average ERP waveform
# average_erp = mne.grand_average(erp_array_target)
# average_erp.plot(ylim=dict(eeg=(-2e7, 2e7)))
#
# # Explore topographical distribution of the average ERP
# # average_erp.plot_topomap(times=[0.1, 0.2, 0.3, 0.4], ch_type='eeg')
#
# plt.show()
#
# # Compute and plot the average ERP waveform
# average_erp = mne.grand_average(erp_array_base)
# average_erp.plot(ylim=dict(eeg=(-2e7, 2e7)))
#
# # Explore topographical distribution of the average ERP
# # average_erp.plot_topomap(times=[0.1, 0.2, 0.3, 0.4], ch_type='eeg')
#
# plt.show()