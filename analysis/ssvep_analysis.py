import glob
import matplotlib.pyplot as plt
import mne

# Specify SSVEP data directory
# E.g. './data/carson/ssvep/*.gdf'
dir = './data/brian/ssvep/*.gdf'

# Find and list SSVEP data files
SSVEPfiles = glob.glob(dir)

# Define EEG channels of interest
eeg_channels = [
    'FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 
    'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 
    'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 
    'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2'
]
eog_channels = ['sens13', 'sens14', 'sens15']
standard_montage = mne.channels.make_standard_montage('standard_1020')

epochs = []
for file in SSVEPfiles:
    # read the raw data
    raw = mne.io.read_raw_gdf(file, eog_channels, preload=True, stim_channel='Status')
    # setting montage for naming
    raw.set_montage(standard_montage, match_case=False)
    # picking only the eeg channels of interest
    raw.pick_channels(eeg_channels)

    # Apply notch filter of 60 Hz
    raw.notch_filter(60, verbose='INFO')

    events, _ = mne.events_from_annotations(raw)

    # Extract 7.5 Hz epochs
    event_id = 2
    freq = 7.5
    epoch = mne.Epochs(raw, events, event_id, event_repeated='merge', tmin=1, tmax=8, baseline=None, verbose='INFO')
    # Plot power spectral densitites (PSDs)
    figure = epoch[f'{event_id}'][0].compute_psd(method='welch', fmin=0, fmax=25, tmin=0, tmax=120).plot()
    figure.axes[0].set_title(f'SSVEP ({freq} Hz)')
    figure.axes[0].axvline(x=freq, color='red', linestyle='--', label='1f')
    figure.axes[0].axvline(x=freq*2, color='blue', linestyle='--', label='2f')

    # Extract 8.57 Hz epochs
    event_id = 3
    freq = 8.57
    epoch = mne.Epochs(raw, events, event_id, event_repeated='merge', tmin=1, tmax=8, baseline=None, verbose='INFO')
    # Plot power spectral densitites (PSDs)
    figure = epoch[f'{event_id}'][0].compute_psd(method='welch', fmin=0, fmax=25, tmin=0, tmax=120).plot()
    figure.axes[0].set_title(f'SSVEP ({freq} Hz)')
    figure.axes[0].axvline(x=freq, color='red', linestyle='--', label='1f')
    figure.axes[0].axvline(x=freq*2, color='blue', linestyle='--', label='2f')

    # Extract 10 Hz epochs
    event_id = 4
    freq = 10
    epoch = mne.Epochs(raw, events, event_id, event_repeated='merge', tmin=1, tmax=8, baseline=None, verbose='INFO')
    # Plot power spectral densitites (PSDs)
    figure = epoch[f'{event_id}'][0].compute_psd(method='welch', fmin=0, fmax=25, tmin=0, tmax=120).plot()
    figure.axes[0].set_title(f'SSVEP ({freq} Hz)')
    figure.axes[0].axvline(x=freq, color='red', linestyle='--', label='1f')
    figure.axes[0].axvline(x=freq*2, color='blue', linestyle='--', label='2f')

    # Extract 12 Hz epochs
    event_id = 5
    freq = 12
    epoch = mne.Epochs(raw, events, event_id, event_repeated='merge', tmin=1, tmax=8, baseline=None, verbose='INFO')
    # Plot power spectral densitites (PSDs)
    figure = epoch[f'{event_id}'][0].compute_psd(method='welch', fmin=0, fmax=25, tmin=0, tmax=120).plot()
    figure.axes[0].set_title(f'SSVEP ({freq} Hz)')
    figure.axes[0].axvline(x=freq, color='red', linestyle='--', label='1f')
    figure.axes[0].axvline(x=freq*2, color='blue', linestyle='--', label='2f')

    plt.show()