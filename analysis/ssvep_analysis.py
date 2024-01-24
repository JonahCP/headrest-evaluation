import mne
import analysis
import matplotlib.pyplot as plt
import scipy.signal
# Load data
# Make sure you use numpy version 1.23.5
raw_data_1 = mne.io.read_raw_gdf('StimuliVerificationTrials/11_17_23_364DEEGtests/JasonSSVEP1.gdf', eog=['sens13', 'sens14', 'sens15'], preload=True, stim_channel='Status')
raw_data_2 = mne.io.read_raw_gdf('StimuliVerificationTrials/11_17_23_364DEEGtests/JasonSSVEP2.gdf', eog=['sens13', 'sens14', 'sens15'], preload=True, stim_channel='Status')
raw_data_3 = mne.io.read_raw_gdf('StimuliVerificationTrials/11_17_23_364DEEGtests/JasonSSVEP3.gdf', eog=['sens13', 'sens14', 'sens15'], preload=True, stim_channel='Status')
raw_data_4 = mne.io.read_raw_gdf('StimuliVerificationTrials/11_17_23_364DEEGtests/JasonSSVEP4.gdf', eog=['sens13', 'sens14', 'sens15'], preload=True, stim_channel='Status')

raw_data_files = [raw_data_1, raw_data_2, raw_data_3, raw_data_4]

# Set montage, aka electrode positions
standard_montage = mne.channels.make_standard_montage('standard_1020')
for raw_data in raw_data_files:
    raw_data.set_montage(standard_montage, match_case=False)

# Filter data
filtered_data = []
for i, raw in enumerate(raw_data_files):
    filtered_data_file = raw.copy()
    # Add bandpass filter between 1 and 30 Hz
    # filtered_data_file = raw.copy().filter(l_freq=1, h_freq=30, verbose='INFO')

    # Add notch filter at 60 Hz
    filtered_data_file.notch_filter(60, verbose='INFO')
    filtered_data.append(filtered_data_file)

# Set montage, aka electrode positions
standard_montage = mne.channels.make_standard_montage('standard_1020')
for raw_data in raw_data_files:
    raw_data.set_montage(standard_montage, match_case=False)

mapping = {
    1: '7.5 Hz',
    2: '8.57 Hz',
    3: '10 Hz',
    4: '12 Hz',
    6: 'START',
    5: 'END'
}

list_of_epochs = []

for data in filtered_data:
    events, event_id = analysis.set_mapping(data, mapping) # type: ignore
    epochs = mne.Epochs(data, events, event_id, event_repeated='merge', tmin=1, tmax=8, baseline=None, verbose='INFO')
    list_of_epochs.append(epochs)

# Plot PSDs
freqs = [7.5, 8.57, 10, 12]
freqs = [8.57]


# https://mne.tools/stable/auto_tutorials/time-freq/50_ssvep.html#sphx-glr-auto-tutorials-time-freq-50-ssvep-py

for i, epochs in enumerate(list_of_epochs):
    for freq in freqs:

        epochs[f"{freq} Hz"][0].compute_psd(method='welch').plot(dB=True)


        # Welch's method is better for unsegemented data
        # It involved segmented the data into smaller chunks and averaging the PSDs
        # window = scipy.signal.get_window('hamming', 3585)
        # epochs[f"{freq} Hz"][0].compute_psd(method='welch', n_fft= 3585, window=window).plot(dB=True)

        # Multitaper is better for segmented data (OK apparently not but it's an alternative to Welch's method)
        # It involves using different windows on the same segment of data and averaging the PSDs
        # epochs[f"{freq} Hz"][0].compute_psd(method='multitaper', adaptive=True, low_bias=True).plot(dB=False)

        plt.title(f"SSVEP{i} {freq}")

plt.show()