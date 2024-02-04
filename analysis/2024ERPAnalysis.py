import os

import mne
import numpy
import numpy as np
from matplotlib import pyplot as plt

# Load the EEG data from the .gdf file
ERPfile1 = 'StimuliVerificationTrials/data2/CarsonERP1.gdf'
ERPfile2 = 'StimuliVerificationTrials/data2/CarsonERP2.gdf'
ERPfile3 = 'StimuliVerificationTrials/data2/CarsonERP3.gdf'
ERPfile4 = 'StimuliVerificationTrials/data2/CarsonERP4.gdf'
ERPfile5 = 'StimuliVerificationTrials/data2/CarsonERP5.gdf'

ERPfile6 = 'StimuliVerificationTrials/data/BrianERP1.gdf'
ERPfile7 = 'StimuliVerificationTrials/data/BrianERP2.gdf'
ERPfile8 = 'StimuliVerificationTrials/data/BrianERP3.gdf'
ERPfile9 = 'StimuliVerificationTrials/data/BrianERP4.gdf'
ERPfile10 = 'StimuliVerificationTrials/data/BrianERP5.gdf'

# Define array of ERP Files
ERPfiles = [ERPfile1, ERPfile2, ERPfile3, ERPfile4, ERPfile5]

# Define EOG channels
eog_channels = ['sens13', 'sens14', 'sens15']

channels_of_interests = ['FP1', 'FPZ', 'FP2', 'F3', 'FZ', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2', 'sens13', 'sens14', 'sens15']

standard_montage = mne.channels.make_standard_montage('standard_1020')

# Initialize an empty list to store ERP data for each channel
list_of_avg_base_stimuli_across_all_trials = []
list_of_avg_target_stimuli_across_all_trials = []

for i, file in enumerate(ERPfiles):
    # Extract only the file name from the full filepath for plot title
    file_name = os.path.basename(file)

    raw = mne.io.read_raw_gdf(file, preload=True, eog=eog_channels)

    # This is just added in when including Brian's files
    if i > 4:
        raw.annotations.description[raw.annotations.description == '10'] = 'temp'
        raw.annotations.description[raw.annotations.description == '20'] = '10'
        raw.annotations.description[raw.annotations.description == 'temp'] = '20'

    # setting montage for naming
    raw.set_montage(standard_montage, match_case=False)

    raw.pick_channels(channels_of_interests)

    # Define events based on your experimental paradigm
    events, event_id = mne.events_from_annotations(raw)

    scalings = dict(eeg=55, eog=75)

    # THIS IS PLOTTING THE RAW DATA, NO FILTER
    # raw.plot(events, scalings=scalings)

    # Where 2 is base and 3 is target stimuli
    event_ids_of_interest = [2, 3]

    # Picks are just defining what type of channels we are using, in this case EEG and EOG
    picks = mne.pick_types(raw.info, eeg=True, eog=True)

    # Create epochs around events of interest (e.g., visual stimuli)
    base_epochs = None
    target_epochs = None

    for eid in event_ids_of_interest:
        final_epochs = mne.Epochs(raw, events, eid, tmin=-0.25, tmax=0.75, reject={'eeg': 150, 'eog': 100}, detrend=1,
                                  preload=True, event_repeated='merge', picks=picks)
        # final_epochs.plot(picks=picks, events=events, scalings=scalings)

        final_epochs.filter(l_freq=.05, h_freq=15, picks='eeg')

        # final_epochs.plot(picks=picks, events=events, scalings=scalings)

        final_epochs = final_epochs.apply_baseline(baseline=(-.25, 0))

        # final_epochs.plot(picks=picks, events=events, scalings=scalings)

        if eid == 2:
            base_epochs = final_epochs.copy()
        else:
            target_epochs = final_epochs.copy()

    base_epochs.plot(picks=picks, events=events, scalings=scalings)
    target_epochs.plot(picks=picks, events=events, scalings=scalings)

    # The following are evoked objects in mne
    base_epochs_avg = base_epochs.average(picks='eeg')
    target_epochs_avg = target_epochs.average(picks='eeg')

    # Add the two evoked objects above to a list to be grand averaged after
    list_of_avg_base_stimuli_across_all_trials.append(base_epochs_avg)
    list_of_avg_target_stimuli_across_all_trials.append(target_epochs_avg)

# base_epochs_avg.plot(picks='eeg')
# target_epochs_avg.plot(picks='eeg')

# Take the grand average of the base and target stimulis
grand_average_base_stimuli = mne.grand_average(list_of_avg_base_stimuli_across_all_trials)
grand_average_target_stimuli = mne.grand_average(list_of_avg_target_stimuli_across_all_trials)

channels_of_interests = ['FP1', 'FPZ', 'FP2', 'F3', 'FZ', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2']

# Plot the average of the base stimuli and target stimuli for each channel
for ch_name in channels_of_interests:
    fig, axes = plt.subplots()
    base_data = grand_average_base_stimuli.get_data(picks=ch_name)
    target_data = grand_average_target_stimuli.get_data(picks=ch_name)
    # Plot the data for the current channel
    axes.plot(grand_average_base_stimuli.times * 1000, base_data.mean(axis=0), color='black', label='Base Stimuli')
    axes.plot(grand_average_target_stimuli.times * 1000, target_data.mean(axis=0), color='red', label='Target Stimuli')

    # Add vertical lines at specific time points
    axes.axvline(x=0, color='blue', linestyle='--', label='Stimuli Shown')
    # axes.axvline(x=250, color='red', linestyle='--', label='Vertical Line at 250ms')
    axes.axvline(x=300, color='green', linestyle='--', label='Vertical Line at 300ms')
    # axes.axvline(x=350, color='red', linestyle='--', label='Vertical Line at 350ms')

    # Set labels and title
    axes.set_title(f'Averaged Epochs for {ch_name}')
    axes.set_xlabel('Time (ms)')
    axes.set_ylabel('Amplitude (uV)')

    # Invert the y-axis to flip the negative values upwards
    plt.gca().invert_yaxis()

    # Add legend
    axes.legend()

    # Show the plot
    plt.show()


grand_average_target_stimuli.plot(picks='eeg')
