import os
import mne
import pandas as pd
from pandas import DataFrame, read_csv
import numpy
import re
import numpy as np
from matplotlib import axis, pyplot as plt
from autoreject import get_rejection_threshold, AutoReject, compute_thresholds

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
        (EventsDataFrame['events'] == '7.50 Hz START'),
        (EventsDataFrame['events'] == '8.57 Hz START'),
        (EventsDataFrame['events'] == '10.00 Hz START'),
        (EventsDataFrame['events'] == '12.00 Hz START'),
        (EventsDataFrame['events'] == '7.50 Hz END'),
        (EventsDataFrame['events'] == '8.57 Hz END'),
        (EventsDataFrame['events'] == '10.00 Hz END'),
        (EventsDataFrame['events'] == '12.00 Hz END'),
    ]

    values = [1, 2, 3, 4, 5, 6, 7, 8, 9]

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

def get_trimmed_files(pick_dirs=None):
    base_directory = 'hr_data'

    directories = [f for f in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, f)) and f in pick_dirs] # type: ignore

    participant_names = []
    all_filepaths = []
    for directory in directories:
        print(f'Checking directory: {directory}')
        if '_' in directory:
            # Split the directory name to get the participant number
            name = directory.split('_')[1]
        else:
            name = (directory[:-1]  + ' ' + directory[-1]).title()

        for i in range(1, 9):
            file_path = f'{base_directory}/{directory}/ssvep{i}_trim.csv'
            if os.path.exists(file_path):
                print(f'Found file: {file_path}')
                participant_names.append(name)
                all_filepaths.append(file_path)

    return all_filepaths, participant_names

def get_timestamps(pick_dirs=None):
    base_directory = 'hr_data'

    directories = [f for f in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, f)) and f in pick_dirs] # type: ignore
    all_timestamps = []

    for directory in directories:
        print(f'Checking directory: {directory}')
        if '_' in directory:
            name = directory.split('_')[1]
        else:
            name = directory

        subject_base_directory = f'{base_directory}/{directory}'

        ssvep_directories = [f for f in os.listdir(subject_base_directory)
                             if os.path.isdir(os.path.join(subject_base_directory, f)) and 'ssvep' in f]

        for ssvep_directory in ssvep_directories:
            ssvep_directory_path = f'{base_directory}/{directory}/{ssvep_directory}'
            ssvep_timestamp = [f for f in os.listdir(ssvep_directory_path) if re.match(r'ssvep_timestamps_.*\.csv', f)][0]
            
            if ssvep_timestamp:
                timestamp_path = f'{base_directory}/{directory}/{ssvep_directory}/{ssvep_timestamp}'
                all_timestamps.append(timestamp_path)

    return all_timestamps

def compute_psd_graph(psds, freqs, freq):
    
    # Plot PSD spectrum
    figure_psd = plt.figure()

    psd_db = 10 * np.log10(psds)

    if len(psds.shape) == 3:
        # Average across channels and trials
        psd_mean = psd_db.mean(axis=(0, 1))
        psd_std = psd_db.std(axis=(0, 1))

        psd_channel_mean = psd_db.mean(axis=(0))

        channel_1 = psd_channel_mean[0, :]
        channel_2 = psd_channel_mean[1, :]
        channel_3 = psd_channel_mean[2, :]
    else:
        # Average across channels
        psd_mean = psd_db.mean(axis=(0))
        psd_std = psd_db.std(axis=(0))

        channel_1 = psd_db[0, :]
        channel_2 = psd_db[1, :]
        channel_3 = psd_db[2, :]

    plt.plot(freqs, channel_1, color='blue', linewidth=0.5, label='Channel 1')
    plt.plot(freqs, channel_2, color='green', linewidth=0.5, label='Channel 2')
    plt.plot(freqs, channel_3, color='red', linewidth=0.5, label='Channel 3')
    plt.plot(freqs, psd_mean, zorder=3, color='black', linewidth=2, label='Average PSD')
    # plt.fill_between(freqs, psd_mean - psd_std, psd_mean + psd_std, color='black', alpha=0.2)

    plt.title(f'PSD ({freq} Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB)')

    # Create vertical lines to indicate harmonics
    plt.axvline(x=freq, color='red', linestyle='--', label='1f')
    plt.axvline(x=freq*2, color='blue', linestyle='--', label='2f')
    # plt.axvline(x=freq*3, color='green', linestyle='--', label='3f')
    plt.legend(loc='upper right')
    plt.ylim(-10, 50)

    return figure_psd

def compute_snr_graph(snr, freqs, freq):
    if len(snr.shape) == 3:
        # Average across trials and channels
        snr_mean = snr.mean(axis=(0, 1))
        snr_std = snr.std(axis=(0, 1))

        # Mean SNR per channel 
        snr_channel_mean = snr.mean(axis=0)

        channel_1 = snr_channel_mean[0, :]
        channel_2 = snr_channel_mean[1, :]
        channel_3 = snr_channel_mean[2, :]
    else:
        # Average across channels
        snr_mean = snr.mean(axis=0)
        snr_std = snr.std(axis=0)

        channel_1 = snr[0, :]
        channel_2 = snr[1, :]
        channel_3 = snr[2, :]

    # Plot SNR spectrum with mean and standard deviation
    figure_snr = plt.figure()

    # Plot channel SNR
    plt.plot(freqs, channel_1, zorder=1, color='blue', linewidth=0.5, label='Channel 1')
    plt.plot(freqs, channel_2, zorder=1, color='green', linewidth=0.5, label='Channel 2')
    plt.plot(freqs, channel_3, zorder=1, color='red', linewidth=0.5, label='Channel 3')

    # Plotting the mean SNR
    plt.plot(freqs, snr_mean, zorder=3, color='black', linewidth=2, label='Average SNR')  
    # plt.fill_between(freqs, snr_mean - snr_std, snr_mean + snr_std, color='black', alpha=0.2, zorder=1)  # Filling between ±1 SD

    plt.title(f'SNR ({freq} Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SNR')

    # Marking stimulus frequency and its harmonics
    plt.axvline(x=freq, color='red', linestyle=':', dashes=[1, 5], label='1f', zorder=0)
    plt.axvline(x=freq*2, color='blue', linestyle=':', dashes=[1, 5], label='2f', zorder=0)
    # plt.axvline(x=freq*3, color='green', linestyle=':', dashes=[1, 5], label='3f', zorder=0)
    plt.ylim(0, 30)

    plt.legend(loc='upper right')

    return figure_snr

def create_snr_bar_graph(snr_values, freqs, name): 
    color_mapping = {
    7.5: 'darkblue',
    8.57: 'darkgreen',
    10: 'darkred',
    12: 'darkorange',
    15: 'mediumblue',
    17.14: 'green',
    20: 'firebrick',
    22.5: 'blue',
    24: 'orange',
    # 25.71: 'seagreen',
    # 30: 'red',
    # 36: 'gold'
    }

    # freq_plot = [7.5, 8.57, 10, 12, 15, 17.14, 20, 22.5, 24, 25.71, 30, 36]
    freq_plot = [7.5, 8.57, 10, 12, 15, 17.14, 20, 22.5, 24]
    stim_freqs = [7.5, 8.57, 10, 12]

    fig, ax = plt.subplots()
    # Change size of the figure
    fig.set_size_inches(15, 6)

    # Adjust the width of the bars and the spacing
    width = 0.05 
    spacing = 0.005

    labels = [f"{f} Hz Trial" for f in stim_freqs]
    x_base = np.arange(len(labels))  # Base x locations for the groups

    for i, f in enumerate(freq_plot):
        color = color_mapping[f]

        # Calculate positions for the bars
        x_positions = x_base + (i - len(freq_plot) / 2) * (width + spacing)

        # Plot bars for each stimulus frequency with calculated positions and colors
        for j, snr in enumerate(snr_values):
            freq_idx = np.argmin(abs(np.array(freqs) - f))
            snr_mean_at_freq = snr[:, :, freq_idx].mean(axis=(0, 1))
            snr_std_at_freq = snr[:, :, freq_idx].std(axis=(0, 1))
            
            ax.bar(
                x_positions[j],  
                snr_mean_at_freq,
                width,
                yerr=snr_std_at_freq,
                color=color,
        label=f"{f:.2f} Hz" if j == 0 else "",
            )

    # Customize plot appearance
    ax.set_ylabel("SNR")
    ax.set_title("Average SNR at target frequencies for each stimulus for " + name)
    ax.set_xticks(x_base)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 50)

    ax.legend(title="Frequency:", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.axhline(1, ls="--", c="r")  # Optional: horizontal line at SNR=1
    plt.tight_layout()

    return fig

# Specify files to use
# us = ['Feb27_Robert', 'Feb27_Jason', 'Feb23_Ella']
us = []
participants = ['participant' + str(i) for i in range(1, 7)] 
pick_dirs = us + participants
# pick_dirs = ['Feb27_Robert', 'Feb27_Jason']

SSVEPFiles, names_files = get_trimmed_files(pick_dirs=pick_dirs)
SSVEPTimestamps = get_timestamps(pick_dirs=pick_dirs)

FileMatrix = [SSVEPFiles, SSVEPTimestamps, names_files]

FileDataFrame = pd.DataFrame(FileMatrix)

name = 'Participants 1-6'

# Create a report for the current file
report = mne.Report(title=f'SSVEP {name} Report', verbose=True)

# Parametesr
freq_mapping = {
    2: 7.5,
    3: 8.57,
    4: 10,
    5: 12
    }

tmin = 1
tmax = 7
n_fft = int(206 * ( tmax - tmin )/ 2)
n_overlap = n_fft // 2
window = 'hamming'

raw_all = []
participant_raw = {}

for i in range(FileDataFrame.columns.size):
    SSVEP_df = read_csv(FileDataFrame[i].iloc[0])
    SSVEP_events_df = read_csv(FileDataFrame[i].iloc[1])
    SSVEP_name = FileDataFrame[i].iloc[2]
    events_info = format_events_frame(SSVEP_events_df)

    # start time of the trial
    start_time = events_info['datetimes'].iloc[0]

    # Pass in the SSVEP_df and the start time of the trial to have an accurate start time for both
    raw = format_raw_data_frame(SSVEP_df, start_time)
    ch_names = ['Channel 1', 'Channel 2', 'Channel 3']
    ch_types = ['eeg', 'eeg', 'eeg']
    sampling_frequency = 206
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sampling_frequency) # type: ignore
    raw = mne.io.RawArray(raw[:3, :], info).set_meas_date(start_time)
    onset_array = events_info['seconds_since_start'].values
    description_array = events_info['TID'].values.astype(int)
    annotations = mne.Annotations(onset=onset_array, description=description_array, duration=0.001953125,
                                  orig_time=start_time)
    raw.set_annotations(annotations) # type: ignore
    raw_all.append(raw)

    if SSVEP_name not in participant_raw:
        participant_raw[SSVEP_name] = []

    participant_raw[SSVEP_name].append(raw.copy()) # type: ignore

raw_all = mne.concatenate_raws(raw_all)
events, _ = mne.events_from_annotations(raw_all)
snr_values = []

# Compute PSD and SNR for all participants
for event_id, freq in freq_mapping.items():
    # Extract epochs for the current frequency
    epochs = mne.Epochs(raw_all, events, event_id, event_repeated='merge', tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=None)

    # Highpass filter at 1 Hz to improve autoreject
    epochs.filter(1, None, fir_design='firwin', verbose=True)
    num_epochs = epochs.__len__()

    original_indices = epochs.selection

    epochs_copy = epochs.copy()

    if epochs.__len__() > 2:
        # Drop bad epochs
        reject = get_rejection_threshold(epochs, cv=epochs.__len__())['eeg']
        epochs.drop_bad(reject={'eeg': reject}) # type: ignore

        if epochs.__len__() == 0:
            print("No epochs found")
            break
    
    epochs_indices = epochs.selection

    # # Determine how many epochs were kept
    kept_epochs = len(epochs_indices)
    html = f'<p> Kept {kept_epochs} epochs out of {num_epochs} for freq {freq} </p>'
    report.add_html(html = html, title='Kept', section='All Participants All Trials')

    # # Look at dropped files
    # epoch_indices_mask = np.isin(original_indices, epochs_indices)
    # epochs_copy.drop(epoch_indices_mask, reason='Dropped trials', verbose=True)
    # epochs = epochs_copy
    # if epochs.__len__() == 0:
    #     print("No epochs found")
    #     break

    spectrums = epochs.compute_psd(method='welch', fmin=1, fmax=40, window=window, n_overlap = n_overlap, n_fft=n_fft)

    psds, freqs = spectrums.get_data(return_freqs=True)
    figure_psd = compute_psd_graph(psds, freqs, freq)
    report.add_figure(figure_psd, f'PSD ({freq} Hz)', section='All Participants All Trials')
    plt.close()

    snr = snr_spectrum(psds, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1)
    snr_values.append(snr)
    figure_snr = compute_snr_graph(snr, freqs, freq)

    report.add_figure(figure_snr, f'SNR ({freq} Hz)', section='All Participants All Trials')
    plt.close()

try: # Try statement is used when plotting dropped trials
    fig = create_snr_bar_graph(snr_values, freqs, name)
    report.add_figure(fig, 'Average SNR at target frequencies', section='All Participants All Trials')
except:
    print("No Dropped Trials")

# Compute PSD and SNR for each participant
for name, raws in participant_raw.items():

    raw_concat = mne.concatenate_raws(raws)
    events, _ = mne.events_from_annotations(raw_concat)
    snr_values = []

    # Compute PSD for each frequency and average
    for event_id, freq in freq_mapping.items():
        # Extract epochs for the current frequency
        epochs = mne.Epochs(raw_concat, events, event_id, event_repeated='merge', tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=None)

        # Apply highpass filter at 1 Hz to improve autoreject
        epochs.filter(1, None, fir_design='firwin', verbose=True)
        num_epochs = epochs.__len__()

        original_indices = epochs.selection

        epochs_copy = epochs.copy()

        if epochs.__len__() > 2:
            reject = get_rejection_threshold(epochs, cv=epochs.__len__())['eeg']
            epochs.drop_bad(reject={'eeg': reject}) # type: ignore

            if epochs.__len__() == 0:
                print("No epochs found")
                break

        epochs_indices = epochs.selection

        # Determine how many epochs were kept
        kept_epochs = len(epochs_indices)

        # Determine trials that were dropped by comparing original indices with new indices
        dropped_trials = numpy.setdiff1d(original_indices, epochs_indices)
        dropped_trials = dropped_trials // 10 + 1
        if dropped_trials.size == 0:
            dropped_trials = 'None'

        html = f"""
        <p> Kept {kept_epochs} epochs out of {num_epochs} for freq {freq}. </p>
        <p> Dropped trials: {dropped_trials} </p>"""
        
        report.add_html(html = html, title='Kept epochs', section=f'{name} All Trials')

        # # Look at dropped files
        # epoch_indices_mask = np.isin(original_indices, epoch_indices)
        # epochs_copy.drop(epoch_indices_mask, reason='Dropped trials', verbose=True)
        # epochs = epochs_copy
        # epoch_indices = epochs.selection
        # if epochs.__len__() == 0:
        #     print("No epochs found")
        #     break

        # Instead compute psds first and then average them
        spectrums = epochs.compute_psd(method='welch', fmin=1, fmax=40, window=window, n_overlap = n_overlap, n_fft=n_fft)

        psds, freqs = spectrums.get_data(return_freqs=True)
        figure_psd = compute_psd_graph(psds, freqs, freq)
        report.add_figure(figure_psd, f'Average PSD ({freq} Hz)', section=f'{name} All Trials')
        plt.close()

        # Compute SNR
        snr = snr_spectrum(psds, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1)
        snr_values.append(snr)
        figure_snr = compute_snr_graph(snr, freqs, freq)

        # Add the figure to the report
        report.add_figure(figure_snr, f'Average SNR ({freq} Hz)', section=f'{name} All Trials')
        plt.close()

        report.add_html(html = html, title='Kept epochs', section=f'{name} {freq} Trials')


        # Compute PSD and SNR per participant and trial
        for i, trial in enumerate(epochs_indices):
            epoch = epochs[i].load_data() # type: ignore
            raw_data = epoch.plot(scalings={'eeg': 500}, show=False)
            report.add_figure(raw_data, f'Raw data for ({freq} Hz, Trial {trial // 10 + 1})', section=f'{name} {freq} Trials')
            plt.close()

            # Compute PSD
            spectrum = epoch.compute_psd(method='welch', fmin=1, fmax=40, window=window, n_overlap = n_overlap, n_fft=n_fft)
            psd, freqs = spectrum.get_data(return_freqs=True)
            figure_psd = compute_psd_graph(psd, freqs, freq)
            report.add_figure(figure_psd, f'PSD ({freq} Hz, Trial {trial // 10 + 1})', section=f'{name} {freq} Trials')
            plt.close()

            # Compute SNR
            snr = snr_spectrum(psd, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1)
            figure_snr = compute_snr_graph(snr, freqs, freq)
            report.add_figure(figure_snr, f'SNR ({freq} Hz, Trial {trial // 10 + 1})', section=f'{name} {freq} Trials')
            plt.close()

    try: # Try statement is used when plotting dropped trials 
        # Plot SNR bar graph
        fig = create_snr_bar_graph(snr_values, freqs, name)
        report.add_figure(fig, f'Average SNR at target frequencies for {name}', section=f'{name} All Trials')
    except:
        print("No Dropped Trials")

report.save(f'./reports/{name}_SSVEP_headrest.html', overwrite=True)