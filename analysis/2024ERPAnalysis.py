import os

import mne
import numpy
import numpy as np
from matplotlib import pyplot as plt

# Load the EEG data from the .gdf file
ERPfile1 = 'eeg_data/carson/erp/CarsonERP1.gdf'
ERPfile2 = 'eeg_data/carson/erp/CarsonERP2.gdf'
ERPfile3 = 'eeg_data/carson/erp/CarsonERP3.gdf'
ERPfile4 = 'eeg_data/carson/erp/CarsonERP4.gdf'
ERPfile5 = 'eeg_data/carson/erp/CarsonERP5.gdf'

ERPfile6 = 'eeg_data/brian/erp/BrianERP1.gdf'
ERPfile7 = 'eeg_data/brian/erp/BrianERP2.gdf'
ERPfile8 = 'eeg_data/brian/erp/BrianERP3.gdf'
ERPfile9 = 'eeg_data/brian/erp/BrianERP4.gdf'
ERPfile10 = 'eeg_data/brian/erp/BrianERP5.gdf'

ERPfile11 = 'eeg_data/jonah/erp/JonahERP1.gdf'
ERPfile12 = 'eeg_data/jonah/erp/JonahERP2.gdf'
ERPfile13 = 'eeg_data/jonah/erp/JonahERP3.gdf'
ERPfile14 = 'eeg_data/jonah/erp/JonahERP4.gdf'
ERPfile15 = 'eeg_data/jonah/erp/JonahERP5.gdf'
ERPfile16 = 'eeg_data/jonah/erp/JonahERP6.gdf'
ERPfile17 = 'eeg_data/jonah/erp/JonahERP7.gdf'
ERPfile18 = 'eeg_data/jonah/erp/JonahERP8.gdf'

ERPfile19 = 'eeg_data/ella/erp/EllaERP1.gdf'
ERPfile20 = 'eeg_data/ella/erp/EllaERP2.gdf'
ERPfile21 = 'eeg_data/ella/erp/EllaERP3.gdf'
ERPfile22 = 'eeg_data/ella/erp/EllaERP4.gdf'
ERPfile23 = 'eeg_data/ella/erp/EllaERP5.gdf'
ERPfile24 = 'eeg_data/ella/erp/EllaERP6.gdf'
ERPfile25 = 'eeg_data/ella/erp/EllaERP7.gdf'
ERPfile26 = 'eeg_data/ella/erp/EllaERP8.gdf'

# Define array of ERP Files
ERPfiles = [ERPfile1, ERPfile2, ERPfile3, ERPfile4, ERPfile5, ERPfile11, ERPfile12, ERPfile13, ERPfile14, ERPfile15, ERPfile16, ERPfile17, ERPfile18, ERPfile19, ERPfile20, ERPfile21, ERPfile22, ERPfile23, ERPfile24, ERPfile25, ERPfile26, ERPfile6, ERPfile7, ERPfile8, ERPfile9, ERPfile10]

# Define EOG channels
eog_channels = ['sens13', 'sens14', 'sens15']

channels_of_interests = ['FP1', 'FPZ', 'FP2', 'F3', 'FZ', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2', 'sens13', 'sens14', 'sens15']

standard_montage = mne.channels.make_standard_montage('standard_1020')

# Initialize an empty list to store ERP data for each channel
list_of_avg_base_stimuli_across_all_trials = []
list_of_avg_target_stimuli_across_all_trials = []

num_of_targets = 0
num_of_base = 0

for i, file in enumerate(ERPfiles):
    # Extract only the file name from the full filepath for plot title
    file_name = os.path.basename(file)

    raw = mne.io.read_raw_gdf(file, preload=True, eog=eog_channels).set_eeg_reference(ref_channels=['O1', 'O2'])

    # raw.compute_psd(method='welch').plot()

    # This is just added in when including Brian's files
    if i > 20:
        raw.annotations.description[raw.annotations.description == '10'] = 'temp'
        raw.annotations.description[raw.annotations.description == '20'] = '10'
        raw.annotations.description[raw.annotations.description == 'temp'] = '20'

    # setting montage for naming
    raw.set_montage(standard_montage, match_case=False)

    raw.pick_channels(channels_of_interests)

    # raw.compute_psd().plot()

    raw.filter(.1, 30, fir_design='firwin')

    # Shows the positions of the electrodes
    # raw.plot_sensors(show_names=True)
    # plt.show()

    # raw.compute_psd().plot()

    # Define events based on your experimental paradigm
    events, event_id = mne.events_from_annotations(raw)

    scalings = dict(eeg=55, eog=75)

    # THIS IS PLOTTING THE RAW DATA, NO FILTER
    # raw.plot(events, scalings=scalings)
    # raw.plot(events, scalings='auto')

    # Where 2 is base and 3 is target stimuli
    event_ids_of_interest = [2, 3]

    # Picks are just defining what type of channels we are using, in this case EEG and EOG
    picks = mne.pick_types(raw.info, eeg=True, eog=True)

    # Create epochs around events of interest (e.g., visual stimuli)
    base_epochs = None
    target_epochs = None

    for eid in event_ids_of_interest:
        final_epochs = mne.Epochs(raw, events, eid, tmin=-0.25, tmax=0.75, reject={'eeg': 150, 'eog': 75}, detrend=1,
                                  preload=True, event_repeated='merge', picks=picks)

        # final_epochs.plot_drop_log()
        # final_epochs.plot(picks=picks, events=events, scalings=scalings)

        # Define filter parameters
        # low_cutoff = 1  # Set your desired low-pass cutoff frequency in Hz
        # high_cutoff = 30  # Set to None for low-pass filtering
        #
        # # Apply low-pass filter
        # final_epochs.filter(low_cutoff, high_cutoff, fir_design='firwin', filter_length=10000)

        # final_epochs.filter(l_freq=.1, h_freq=30, picks='eeg')

        # final_epochs.plot(picks=picks, events=events, scalings=scalings)
        if final_epochs.__len__() == 0:
            break

        final_epochs = final_epochs.apply_baseline(baseline=(-.25, 0))

        # final_epochs.plot(picks=picks, events=events, scalings=scalings)

        if eid == 2:
            base_epochs = final_epochs.copy()
            num_of_base += base_epochs.events.shape[0]
        else:
            target_epochs = final_epochs.copy()
            num_of_targets += target_epochs.events.shape[0]

    # base_epochs.plot(picks=picks, events=events, scalings=scalings)
    # target_epochs.plot(picks=picks, events=events, scalings=scalings)

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

grand_average_base_stimuli.plot_joint()
grand_average_target_stimuli.plot_joint()

# grand_average_base_stimuli.plot_image()
# grand_average_target_stimuli.plot_image()

# report = mne.Report(title="Evoked ERP")
# report.add_evokeds(
#     evokeds=[grand_average_base_stimuli, grand_average_target_stimuli],
#     titles=["Base Stimuli", "Target Stimuli"],
# )
#
# report.save(overwrite=True)


channels_of_interests = ['FP1', 'FPZ', 'FP2', 'F3', 'FZ', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2']

# Calculate the number of rows and columns for the grid
num_rows = 3
num_cols = 3

# Calculate the number of subplots needed
num_subplots = int(np.ceil(len(channels_of_interests) / (num_rows * num_cols)))

# Determine y-axis limits across all channels
min_y = float('inf')
max_y = float('-inf')

for ch_name in channels_of_interests:
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
    for idx, ch_name in enumerate(channels_of_interests[start_channel_idx:end_channel_idx]):
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
        axes[idx].set_title(f'Averaged Epochs for {ch_name}')
        axes[idx].set_xlabel('Time (ms)')
        axes[idx].set_ylabel('Amplitude (uV)')

        # Invert the y-axis to flip the negative values upwards
        axes[idx].invert_yaxis()

        # Set y-axis limits
        axes[idx].set_ylim(min_y, max_y)

        # Add legend
        axes[idx].legend()

    # Adjust layout with auto spacing
    plt.tight_layout()

    # Show the plot
    plt.show()