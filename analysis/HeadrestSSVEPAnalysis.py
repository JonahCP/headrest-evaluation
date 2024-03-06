import os
import mne
import pandas as pd
from pandas import DataFrame, read_csv
import numpy
import re
import numpy as np
from matplotlib import pyplot as plt

def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise

def format_events_frame(EventsDataFrame):
    # Drop the first column since it's the same as the index
    EventsDataFrame = EventsDataFrame.drop(EventsDataFrame.columns[0], axis=1)

    # Convert 'datetimes' column to datetime format
    EventsDataFrame['datetimes'] = pd.to_datetime(EventsDataFrame['datetimes'], format='%Y-%m-%d %H.%M.%S.%f',
                                                  errors='coerce', utc=True)

    # Gets the seconds that have passed since the trial started
    EventsDataFrame['seconds_since_start'] = (
            EventsDataFrame['datetimes'] - EventsDataFrame['datetimes'].min()).dt.total_seconds()

    # Create 'TID' column based on conditions
    conditions = [
        (EventsDataFrame['events'] == 'START') | (EventsDataFrame['events'] == 'END'),
        (EventsDataFrame['events'] == '7.50 Hz START') | (EventsDataFrame['events'] == '7.50 Hz END'),
        (EventsDataFrame['events'] == '8.57 Hz START') | (EventsDataFrame['events'] == '8.57 Hz END'),
        (EventsDataFrame['events'] == '10.00 Hz START') | (EventsDataFrame['events'] == '10.00 Hz END'),
        (EventsDataFrame['events'] == '12.00 Hz START') | (EventsDataFrame['events'] == '12.00 Hz END'),
    ]

    values = [1, 2, 3, 4, 5]

    # Adds the TID next to the events
    EventsDataFrame['TID'] = np.select(conditions, values, default=np.nan)

    # Reorder columns
    EventsDataFrame = EventsDataFrame[
        ['datetimes', 'seconds_since_start', 'events', 'TID'] + [col for col in EventsDataFrame.columns if
                                                                 col not in ['datetimes', 'seconds_since_start',
                                                                             'events', 'TID']]]

    return EventsDataFrame

def format_raw_data_frame(DataFrame, startTime):
    # Convert 'datetime' column to datetime format
    DataFrame['datetime'] = pd.to_datetime(DataFrame['datetime'], format='%Y-%m-%d %H.%M.%S.%f', errors='coerce',
                                           utc=True)

    # Gets the seconds that have passed since the trial started
    DataFrame['seconds_since_start'] = (
            DataFrame['datetime'] - startTime).dt.total_seconds()

    # Get raw channel data
    raw_channel_data = DataFrame[['chnl-1-raw', 'chnl-2-raw', 'chnl-3-raw', 'datetime', 'seconds_since_start']].values.T

    return raw_channel_data

def get_trimmed_files():
    base_directory = 'hr_data'
    # The line below searches the hr_data directory for other directories (which contain the test data)
    directories = [f for f in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, f))]

    files = []

    for directory in directories:
        for i in range(1, 9):
            file_path = f'{base_directory}/{directory}/ssvep{i}_trim.csv'
            files.append(file_path)

    return files

def get_timestamps():
    base_directory = 'hr_data'
    # The line below searches the hr_data directory for other directories (which contain the test data)
    directories = [f for f in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, f))]

    files = []

    for directory in directories:
        subject_base_directory = f'{base_directory}/{directory}'
        # Does the same as before except only grabs the erp directories for a single subject
        ssvep_directories = [f for f in os.listdir(subject_base_directory)
                           if os.path.isdir(os.path.join(subject_base_directory, f)) and 'ssvep' in f]

        for ssvep_directory in ssvep_directories:
            ssvep_directory_path = f'{base_directory}/{directory}/{ssvep_directory}'
            ssvep_timestamp = [f for f in os.listdir(ssvep_directory_path) if re.match(r'ssvep_timestamps_.*\.csv', f)][0]

            files.append(f'{base_directory}/{directory}/{ssvep_directory}/{ssvep_timestamp}')

    return files


TrimmedSSVEPFiles = get_trimmed_files()[16:]
SSVEPTimestamps = get_timestamps()[16:]

FileMatrix = [TrimmedSSVEPFiles, SSVEPTimestamps]

FileDataFrame = pd.DataFrame(FileMatrix)

name = 'Ella Headrest SSVEP Test'

# Create a report for the current file
report = mne.Report(title=f'SSVEP {name} Report', verbose=True)

freq_mapping = {
    2: 7.5,
    3: 8.57,
    4: 10,
    5: 12
    }

tmin = 1
tmax = 7.5

# for i in range(FileDataFrame.columns.size):
for i in range(8):
    SSVEP_df = read_csv(FileDataFrame[i].iloc[0])
    SSVEP_events_df = read_csv(FileDataFrame[i].iloc[1])

    events_info = format_events_frame(SSVEP_events_df)

    # start time of the trial
    start_time = events_info['datetimes'].iloc[0]

    # Pass in the SSVEP_df and the start time of the trial to have an accurate start time for both
    raw = format_raw_data_frame(SSVEP_df, start_time)

    ch_names = ['Channel 1', 'Channel 2', 'Channel 3']

    ch_types = ['eeg', 'eeg', 'eeg']

    sampling_frequency = 206

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sampling_frequency)

    raw = mne.io.RawArray(raw[:3, :], info).set_meas_date(start_time)

    onset_array = events_info['seconds_since_start'].values

    description_array = events_info['TID'].values.astype(int)

    annotations = mne.Annotations(onset=onset_array, description=description_array, duration=0.001953125,
                                  orig_time=start_time)

    raw.set_annotations(annotations)

    # Begin analysis
    raw.pick_channels(ch_names)

    # # Notch filter to remove 60 Hz noise
    raw.notch_filter(60, verbose='INFO')

    # Define events based on your experimental paradigm
    events, event_id = mne.events_from_annotations(raw)

    # Picks are just defining what type of channels we are using, in this case EEG and EOG
    picks = mne.pick_types(raw.info, eeg=True)

    events, _ = mne.events_from_annotations(raw)

    # Save the raw data plot to the report
    report.add_raw(raw, title=f'Raw Data Trial {i+1}', scalings={'eeg': 500})
    
    for event_id, freq in freq_mapping.items():
        # Extract epochs for the current frequency
        epoch = mne.Epochs(raw, events, event_id, event_repeated='merge', tmin=tmin, tmax=tmax, baseline=None, verbose='INFO')
        
        spectrum = epoch[f'{event_id}'][0].compute_psd(method='welch', average='mean', n_fft= int((sampling_frequency * (tmax - tmin))), fmin=1, fmax=40, n_overlap=(int((sampling_frequency * (tmax - tmin)) // 2)))

        figure = spectrum.plot(amplitude=False, average=True, show=False)

        figure.axes[0].set_title(f'SSVEP ({freq} Hz)')
        figure.axes[0].axvline(x=freq, color='red', linestyle='--', label='1f')
        figure.axes[0].axvline(x=freq*2, color='blue', linestyle='--', label='2f')
        figure.axes[0].axvline(x=freq*3, color='green', linestyle='--', label='3f')
        figure.legend(loc='lower left')

        # Add the figure to the report
        report.add_figure(figure, f'SSVEP ({freq} Hz)', section=f'SSVEP Trial {i+1}')
        plt.close()

        # Average PSD across epochs (only have one)
        psd, freqs = spectrum.get_data(return_freqs=True)
        psd_mean = psd.mean(axis=0)

        # Compute SNR
        snr = snr_spectrum(psd_mean, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1)

        print("SNR shape: ", snr.shape)

        # Average across epochs (only have one)
        snr_mean = snr.mean(axis=0)
        snr_std = snr.std(axis=0)

        # Plot SNR spectrum with mean and standard deviation
        figure_snr = plt.figure()
        plt.plot(freqs, snr_mean, zorder=2, color='black')  # Plotting the mean SNR
        plt.fill_between(freqs, snr_mean - snr_std, snr_mean + snr_std, color='black', alpha=0.2, zorder=1)  # Filling between ±1 SD

        plt.title(f'SNR ({freq} Hz)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('SNR')

        # Marking stimulus frequency and its harmonics
        plt.axvline(x=freq, color='red', linestyle=':', dashes=[1, 5], label='1f', zorder=0)
        plt.axvline(x=freq*2, color='blue', linestyle=':', dashes=[1, 5], label='2f', zorder=0)
        plt.axvline(x=freq*3, color='green', linestyle=':', dashes=[1, 5], label='3f', zorder=0)
        plt.ylim(0, 30)

        plt.legend()
        # Add the figure to the report
        report.add_figure(figure_snr, f'SNR ({freq} Hz)', section=f'SSVEP Trial {i+1}')
        plt.close()

report.save(f'./reports/headrest/{name}.html', overwrite=True)
    