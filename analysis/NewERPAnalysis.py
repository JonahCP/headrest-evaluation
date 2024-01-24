import mne
import matplotlib.pyplot as plt

# Load the EEG data from the .gdf file
ERPfile1 = 'StimuliVerificationTrials/data/BrianERP1.gdf'
ERPfile2 = 'StimuliVerificationTrials/data/BrianERP2.gdf'
ERPfile3 = 'StimuliVerificationTrials/data/BrianERP3.gdf'
ERPfile4 = 'StimuliVerificationTrials/data/BrianERP4.gdf'
ERPfile5 = 'StimuliVerificationTrials/data/BrianERP5.gdf'

# Define array of files
ERPfiles = [ERPfile3, ERPfile4, ERPfile5]

# Define EEG channels of interest
# eeg_channels = ['M1', 'T7', 'C3', 'CZ', 'C4', 'T8', 'M2', 'CP5', 'CP6', 'P7', 'P8']
eeg_channels = ['FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'CZ', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2']
# Define EOG channels
eog_channels = ['sens13', 'sens14', 'sens15']

# Initialize an empty array to store ERP data
erp_array_target = []
erp_array_base = []

standard_montage = mne.channels.make_standard_montage('standard_1020')

for file in ERPfiles:
    # read the raw data
    raw = mne.io.read_raw_gdf(file, preload=True, eog=eog_channels)
    # setting montage for naming
    raw.set_montage(standard_montage, match_case=False)
    # picking only the eeg channels of interest
    raw.pick_channels(eeg_channels)

    # Apply the notch filter
    freq_to_notch = 60  # Replace with your desired notch frequency
    raw.notch_filter(freq_to_notch, picks='eeg')

    # Apply bandpass filter between 1 and 30 Hz
    raw.filter(l_freq=1, h_freq=30, picks='eeg')

    # Define events (assuming you have markers for the visual stimuli)
    events, _ = mne.events_from_annotations(raw)

    # Epoch the data around the events
    event_id = 2  # Change 1 to the marker value for visual stimuli
    epochs = mne.Epochs(raw, events, event_id, tmin=-0.25, tmax=0.8, baseline=(-0.25, -0.05), detrend=1, preload=True)

    # Plot the average ERP before artifact rejection
    erp = epochs.average()
    erp_array_target.append(erp)
    erp.plot(ylim=dict(eeg=(-2e7, 2e7)))  # Set y-axis scale to 1e-6
    # erp.plot()

    # Explore topographical distribution before artifact rejection
    # erp.plot_topomap(times=[0.1, 0.2, 0.3, 0.4], ch_type='eeg')

    plt.show()

    # Epoch the data around the events
    event_id = 3  # Change 1 to the marker value for visual stimuli
    epochs = mne.Epochs(raw, events, event_id, tmin=-0.25, tmax=0.8, baseline=(-0.25, -0.05), detrend=1, preload=True)

    # Plot the average ERP before artifact rejection
    erp = epochs.average()
    erp_array_base.append(erp)
    erp.plot(ylim=dict(eeg=(-2e7, 2e7)))  # Set y-axis scale to 1e-6
    # erp.plot()

    # Explore topographical distribution before artifact rejection
    # erp.plot_topomap(times=[0.1, 0.2, 0.3, 0.4], ch_type='eeg')

    plt.show()

# Compute and plot the average ERP waveform
average_erp = mne.grand_average(erp_array_target)
average_erp.plot(ylim=dict(eeg=(-2e7, 2e7)))

# Explore topographical distribution of the average ERP
# average_erp.plot_topomap(times=[0.1, 0.2, 0.3, 0.4], ch_type='eeg')

plt.show()

# Compute and plot the average ERP waveform
average_erp = mne.grand_average(erp_array_base)
average_erp.plot(ylim=dict(eeg=(-2e7, 2e7)))

# Explore topographical distribution of the average ERP
# average_erp.plot_topomap(times=[0.1, 0.2, 0.3, 0.4], ch_type='eeg')

plt.show()