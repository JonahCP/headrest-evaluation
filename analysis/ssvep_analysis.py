import os
import glob
import re
import os
import pandas as pd
import mne
from pandas import DataFrame, read_csv
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

def find_ssvep_files(base_dir, subject_names, folders=['ssvep', 'scrap']):
    """
    Finds and sorts SSVEP files for a given subject and specified folders based on the numbers in the filenames.

    Parameters:
    - base_dir (str): Base directory where subject folders are located.
    - subject_name (str): Name of the subject.
    - folders (list): List of folder names to search within the subject's directory.

    Returns:
    - list: A sorted list of file paths to the SSVEP files.
    """
    all_ssvep_files = {}
    for name in subject_names:
        all_ssvep_files[name] = []
        ssvep_files = []
        for folder in folders:
            path_pattern = os.path.join(base_dir, name, folder, '*SSVEP*.gdf')
            found_files = glob.glob(path_pattern)
            # If file doesn't have a number in the filename, skip it
            found_files = [x for x in found_files if re.search(r'SSVEP(\d+)', x)]
            ssvep_files.extend(found_files)
        # Sort files based on the number in the filename
        ssvep_files_sorted = sorted(ssvep_files, key=lambda x: int(re.search(r'SSVEP(\d+)', x).group(1)))
        ssvep_files_sorted = [x.replace('\\', '/') for x in ssvep_files_sorted]
        all_ssvep_files[name] = ssvep_files_sorted
    return all_ssvep_files

def compute_psd_graph(psds, freqs, freq):
    
    # Plot PSD spectrum
    figure_psd = plt.figure()

    psd_db = 10 * np.log10(psds)

    if len(psds.shape) == 3:
        # Average across channels and trials
        psd_mean = psd_db.mean(axis=(0, 1))
        psd_std = psd_db.std(axis=(0, 1))

        psd_channel_mean = psd_db.mean(axis=(0))

    else:
        # Average across channels
        psd_mean = psd_db.mean(axis=(0))
        psd_std = psd_db.std(axis=(0))

    num_channels = psd_channel_mean.shape[0]

    # Choose a colormap
    cmap = plt.cm.get_cmap('hsv')  # This is a nice perceptually uniform colormap with a good range

    # Generate a list of colors from the colormap
    colors = [cmap(i) for i in np.linspace(0, 1, num_channels)]

    # # Plot each channel with a color from the gradient
    # for channel in range(num_channels):
    #     plt.plot(freqs, psd_channel_mean[channel, :], color=colors[channel], linewidth=0.5, alpha=0.5)
    
    plt.plot(freqs, psd_mean, zorder=3, color='black', linewidth=2, label='Average PSD')
    plt.fill_between(freqs, psd_mean - psd_std, psd_mean + psd_std, color='black', alpha=0.2)

    plt.title(f'Average PSD Across Participants and Occipital Channels ({freq} Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/frequency (dB/Hz)')

    # Create vertical lines to indicate harmonics
    plt.axvline(x=freq, color='red', linestyle='--', label='1f')
    plt.axvline(x=freq*2, color='blue', linestyle='--', label='2f')
    plt.axvline(x=freq*3, color='green', linestyle='--', label='3f')
    plt.legend(loc='upper right')
    plt.ylim(-25, 25)

    return figure_psd

def compute_snr_graph(snr, freqs, freq):
    if len(snr.shape) == 3:
        # Average across trials and channels
        snr_mean = snr.mean(axis=(0, 1))
        snr_std = snr.std(axis=(0, 1))

        # Mean SNR per channel 
        snr_channel_mean = snr.mean(axis=0)

    else:
        # Average across channels
        snr_mean = snr.mean(axis=0)
        snr_std = snr.std(axis=0)

    # Plot SNR spectrum with mean and standard deviation
    figure_snr = plt.figure()

    num_channels = snr_channel_mean.shape[0]

    # Choose a colormap
    cmap = plt.cm.get_cmap('hsv')  # This is a nice perceptually uniform colormap with a good range

    # Generate a list of colors from the colormap
    colors = [cmap(i) for i in np.linspace(0, 1, num_channels)]

    # Plot each channel with a color from the gradient
    for channel in range(num_channels):
        plt.plot(freqs, snr_channel_mean[channel, :], color=colors[channel], linewidth=0.5, alpha=0.5)

    # plt.plot(freqs, channel_1, zorder=1, color='blue', linewidth=0.5, label='Channel 1')
    # plt.plot(freqs, channel_2, zorder=1, color='green', linewidth=0.5, label='Channel 2')
    # plt.plot(freqs, channel_3, zorder=1, color='red', linewidth=0.5, label='Channel 3')

    # Plotting the mean SNR
    plt.plot(freqs, snr_mean, zorder=3, color='black', linewidth=2, label='Average SNR')  
    plt.fill_between(freqs, snr_mean - snr_std, snr_mean + snr_std, color='black', alpha=0.2, zorder=1)  # Filling between ±1 SD

    plt.title(f'SNR ({freq} Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SNR')

    # Marking stimulus frequency and its harmonics
    plt.axvline(x=freq, color='red', linestyle=':', dashes=[1, 5], label='1f', zorder=0)
    plt.axvline(x=freq*2, color='blue', linestyle=':', dashes=[1, 5], label='2f', zorder=0)
    plt.axvline(x=freq*3, color='green', linestyle=':', dashes=[1, 5], label='3f', zorder=0)
    plt.ylim(0, 80)

    plt.legend(loc='upper right')

    return figure_snr

def create_snr_bar_graph(snr_values, freqs, name): 
    color_mapping = {
    7.5: 'royalblue',
    8.57: 'limegreen',
    10: 'red',
    12: 'darkviolet',
    15: 'royalblue',
    17.14: 'limegreen',
    20: 'red',
    22.5: 'royalblue',
    24: 'darkviolet',
    25.71: 'limegreen',
    30: 'red',
    36: 'darkviolet'
    }

    freq_plot = [7.5, 8.57, 10, 12, 15, 17.14, 20, 22.5, 24, 25.71, 30, 36]
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
    ax.set_ylim(0, 80)

    ax.legend(title="Frequency:", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.axhline(1, ls="--", c="r")  # Optional: horizontal line at SNR=1
    plt.tight_layout()

    return fig

base_dir = './eeg_data'
subject_name = ['Brian', 'Ella', 'Jason2', 'Jonah']
# subject_name = ['Jason2']
# ssvep_files = find_ssvep_files(base_dir, subject_name, folders=['ssvep', 'scrap'])
# ssvep_files = find_ssvep_files(base_dir, subject_name, folders=['scrap'])
ssvep_files = find_ssvep_files(base_dir, subject_name, folders=['ssvep'])

# All channels
# eeg_channels = ['FP1', 'FPZ', 'FP2', 'F3', 'FZ', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ', 'O1', 'OZ', 'O2']


# # All near occipital region
# eeg_channels = [
#     'P3', 'PZ', 'P4', 'POZ', 'O1', 'OZ', 'O2'
# ]

eeg_channels = [
    'PZ', 'POZ', 'O1', 'OZ', 'O2'
]

eog_channels = ['sens13', 'sens14', 'sens15']
standard_montage = mne.channels.make_standard_montage('standard_1020')

report_name = 'Breljajo No Scrap POSTER Plots'

# Create a report for the current file
report = mne.Report(title=f'SSVEP {report_name} Report', verbose=True)

# Load raw data for each participant
participant_data = {}

for name, files in ssvep_files.items():
    participant_data[name] = []
    for file in files:
        raw = mne.io.read_raw_gdf(file, eog=eog_channels, preload=True, stim_channel='Status')
        raw.set_montage(standard_montage, match_case=False)
        raw.pick_channels(eeg_channels)
        raw.filter(1, None, fir_design='firwin', verbose=False)
        participant_data[name].append(raw.copy()) # Copy to avoid errors with concatenate

# Load raw data for all participants
raw_data = []
for name, files in ssvep_files.items():
    for file in files:
        raw = mne.io.read_raw_gdf(file, eog=eog_channels, preload=True, stim_channel='Status')
        raw.set_montage(standard_montage, match_case=False)
        raw.pick_channels(eeg_channels)
        raw.filter(1, None, fir_design='firwin', verbose=False)
        raw_data.append(raw)

raw_all = mne.concatenate_raws(raw_data)
events, _ = mne.events_from_annotations(raw_all)

# Parameters
freq_mapping = {
    2: 7.5,
    3: 8.57,
    4: 10,
    5: 12
    }

tmin = 1
tmax = 7
n_fft = int(512 * ( tmax - tmin ))
n_overlap = n_fft // 2
window = 'hamming'

# Compute PSD and SNR for all participants
snr_values = []
for event_id, freq in freq_mapping.items():
    # Extract epochs for the current frequency
    epochs = mne.Epochs(raw_all, events, event_id, event_repeated='merge', tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=None)

    epochs = epochs[::2]

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
    html = f"""<p> Kept {kept_epochs} epochs out of {num_epochs} for freq {freq} </p>
    <p> Rejection Threshold: {reject} </p>
    """
    report.add_html(html = html, title='Kept', section='All Participants All Trials')


    # # Look at dropped files
    # epoch_indices_mask = np.isin(original_indices, epochs_indices)
    # epochs_copy.drop(epoch_indices_mask, reason='Dropped trials', verbose=True)
    # epochs = epochs_copy
    # if epochs.__len__() == 0:
    #     print("No epochs found")
    #     break

    spectrums = epochs.compute_psd(method='welch', fmin=1, fmax=40, window=window, n_overlap = n_overlap, n_fft=n_fft)

    # spectrums.pick('eeg').plot_topo()

    psds, freqs = spectrums.get_data(return_freqs=True)
    figure_psd = compute_psd_graph(psds, freqs, freq)
    report.add_figure(figure_psd, f'PSD ({freq} Hz)', section='All Participants All Trials')

    plt.close()

    # snr = snr_spectrum(psds, noise_n_neighbor_freqs=4, noise_skip_neighbor_freqs=1)
    # snr_values.append(snr)
    # figure_snr = compute_snr_graph(snr, freqs, freq)

    # report.add_figure(figure_snr, f'SNR ({freq} Hz)', section='All Participants All Trials')
    # plt.close()

# try: # Try statement is used when plotting dropped trials
#     fig = create_snr_bar_graph(snr_values, freqs, name)
#     report.add_figure(fig, 'Average SNR at target frequencies', section='All Participants All Trials')
# except:
#     print("No Dropped Trials")

# # Compute PSD and SNR for each participant
# for name, raws in participant_data.items():

#     raw_concat = mne.concatenate_raws(raws)
#     events, _ = mne.events_from_annotations(raw_concat)
#     snr_values = []

#     # Compute PSD for each frequency and average
#     for event_id, freq in freq_mapping.items():
#         # Extract epochs for the current frequency
#         epochs = mne.Epochs(raw_concat, events, event_id, event_repeated='merge', tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=None)

#         epochs = epochs[::2]

#         # Apply highpass filter at 1 Hz to improve autoreject
#         epochs.filter(1, None, fir_design='firwin', verbose=True)
#         num_epochs = epochs.__len__()

#         original_indices = epochs.selection

#         epochs_copy = epochs.copy()

#         if epochs.__len__() > 2:
#             reject = get_rejection_threshold(epochs, cv=epochs.__len__())['eeg']
#             epochs.drop_bad(reject={'eeg': reject}) # type: ignore

#             if epochs.__len__() == 0:
#                 print("No epochs found")
#                 break

#         epochs_indices = epochs.selection

#         # Determine how many epochs were kept
#         kept_epochs = len(epochs_indices)

#         # Determine trials that were dropped by comparing original indices with new indices
#         dropped_trials = np.setdiff1d(original_indices, epochs_indices)
#         dropped_trials = dropped_trials // 10 + 1
#         if dropped_trials.size == 0:
#             dropped_trials = 'None'

#         html = f"""
#         <p> Kept {kept_epochs} epochs out of {num_epochs} for freq {freq}. </p>
#         <p> Dropped trials: {dropped_trials} </p>"""
        
#         report.add_html(html = html, title='Kept epochs', section=f'{name} All Trials')

#         # # Look at dropped files
#         # epoch_indices_mask = np.isin(original_indices, epoch_indices)
#         # epochs_copy.drop(epoch_indices_mask, reason='Dropped trials', verbose=True)
#         # epochs = epochs_copy
#         # epoch_indices = epochs.selection
#         # if epochs.__len__() == 0:
#         #     print("No epochs found")
#         #     break

#         # Instead compute psds first and then average them
#         spectrums = epochs.compute_psd(method='welch', fmin=1, fmax=40, window=window, n_overlap = n_overlap, n_fft=n_fft)

#         psds, freqs = spectrums.get_data(return_freqs=True)
#         figure_psd = compute_psd_graph(psds, freqs, freq)
#         report.add_figure(figure_psd, f'Average PSD ({freq} Hz)', section=f'{name} All Trials')
#         plt.close()

#         # Compute SNR
#         snr = snr_spectrum(psds, noise_n_neighbor_freqs=4, noise_skip_neighbor_freqs=1)
#         snr_values.append(snr)
#         figure_snr = compute_snr_graph(snr, freqs, freq)

#         # Add the figure to the report
#         report.add_figure(figure_snr, f'Average SNR ({freq} Hz)', section=f'{name} All Trials')
#         plt.close()

#         report.add_html(html = html, title='Kept epochs', section=f'{name} {freq} Trials')


    #     # Compute PSD and SNR per participant and trial
    #     for i, trial in enumerate(epochs_indices):
    #         epoch = epochs[i].load_data() # type: ignore
    #         raw_data = epoch.plot(scalings={'eeg': 55}, show=False, picks=['O1', 'OZ', 'O2'])
    #         report.add_figure(raw_data, f'Raw data for ({freq} Hz, Trial {trial // 10 + 1})', section=f'{name} {freq} Trials')
    #         plt.close()

    #         # Compute PSD
    #         spectrum = epoch.compute_psd(method='welch', fmin=1, fmax=40, window=window, n_overlap = n_overlap, n_fft=n_fft)
    #         psd, freqs = spectrum.get_data(return_freqs=True)
    #         figure_psd = compute_psd_graph(psd, freqs, freq)
    #         report.add_figure(figure_psd, f'PSD ({freq} Hz, Trial {trial // 10 + 1})', section=f'{name} {freq} Trials')
    #         plt.close()

    #         # Compute SNR
    #         snr = snr_spectrum(psd, noise_n_neighbor_freqs=4, noise_skip_neighbor_freqs=1)
    #         figure_snr = compute_snr_graph(snr, freqs, freq)
    #         report.add_figure(figure_snr, f'SNR ({freq} Hz, Trial {trial // 10 + 1})', section=f'{name} {freq} Trials')
    #         plt.close()

    # try: # Try statement is used when plotting dropped trials 
    #     # Plot SNR bar graph
    #     fig = create_snr_bar_graph(snr_values, freqs, name)
    #     report.add_figure(fig, f'Average SNR at target frequencies for {name}', section=f'{name} All Trials')
    # except:
    #     print("No Dropped Trials")

# Insert underscores into report_name
report_name = report_name.replace(" ", "_")

report.save(f'./reports/eeg/{report_name}_SSVEP_EEG.html', overwrite=True)