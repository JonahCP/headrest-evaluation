import mne
from matplotlib import pyplot as plt

# Load the EEG data from the .gdf file
ERPfile1 = 'StimuliVerificationTrials/data/BrianERP3.gdf'
ERPfile2 = 'StimuliVerificationTrials/data/BrianERP4.gdf'
ERPfile3 = 'StimuliVerificationTrials/data/BrianERP5.gdf'

# Define EEG channels of interest
eeg_channels = ['CZ']
# Define EOG channels
eog_channels = ['sens13', 'sens14', 'sens15']

standard_montage = mne.channels.make_standard_montage('standard_1020')

raw = mne.io.read_raw_gdf(ERPfile1, preload=True, eog=eog_channels)

# setting montage for naming
raw.set_montage(standard_montage, match_case=False)

# Define events based on your experimental paradigm
events, event_id = mne.events_from_annotations(raw)

# Create epochs around events of interest (e.g., visual stimuli)
epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), detrend=1, preload=True, event_repeated='drop')

# Create EOG epochs to identify EOG artifacts
eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-0.5, -0.2))

# Plot EOG epochs to visually inspect and mark bad segments
eog_epochs.plot_image(combine='mean')

eog_epochs.average().plot_joint()

eog_events = mne.preprocessing.find_eog_events(raw)
onsets = eog_events[:, 0] / raw.info["sfreq"] - 0.25
durations = [0.5] * len(eog_events)
descriptions = ["bad blink"] * len(eog_events)
blink_annot = mne.Annotations(
    onsets, durations, descriptions, orig_time=raw.info["meas_date"]
)
raw.set_annotations(blink_annot)

scalings = dict(eeg=55, eog=75)

raw.plot(events ,scalings=scalings)

plt.show()

test = 3 + 2
