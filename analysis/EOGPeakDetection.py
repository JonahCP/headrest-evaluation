import os
import mne
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Load the EEG data from the .gdf file
ERPfile1 = 'StimuliVerificationTrials/data/BrianERP1.gdf'
ERPfile2 = 'StimuliVerificationTrials/data/BrianERP2.gdf'
ERPfile3 = 'StimuliVerificationTrials/data/BrianERP3.gdf'
ERPfile4 = 'StimuliVerificationTrials/data/BrianERP4.gdf'
ERPfile5 = 'StimuliVerificationTrials/data/BrianERP5.gdf'

# Define array of files
ERPfiles = [ERPfile3, ERPfile4, ERPfile5]

# Define EEG channels of interest
eeg_channels = ['FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'CZ', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2']
# Define EOG channels
eog_channels = ['sens13', 'sens14', 'sens15']

# Read the raw data
raw = mne.io.read_raw_gdf(ERPfiles[0], preload=True, eog=eog_channels)

# Apply the notch filter
freq_to_notch = 60  # Replace with your desired notch frequency
raw.notch_filter(freq_to_notch, picks='eeg')

# Apply bandpass filter between 1 and 30 Hz
raw.filter(l_freq=1, h_freq=30, picks='eeg')

# Extract EOG Data
eog_data, times = raw[eog_channels, :]

# Plot EOG Data
plt.figure()
plt.plot(times, eog_data.T)
plt.title('EOG Channels')
plt.xlabel('Time (s)')
plt.ylabel('EOG Amplitude (uV)')
plt.show()

# Identify Peaks
peaks, _ = find_peaks(eog_data.T[0], height=0)  # Adjust the height parameter as needed

# Calculate Average Peak Size
peak_sizes = eog_data.T[0][peaks]
average_peak_size = np.mean(np.abs(peak_sizes))
print(f'Average Peak Size: {average_peak_size} uV')
