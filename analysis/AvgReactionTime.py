import os
import pandas as pd
from pandas import read_csv
import re

def get_timestamps(i):
    base_directory = f'hr_data/participant{i}'
    # The line below searches the hr_data directory for other directories (which contain the test data)
    directories = [f for f in os.listdir(base_directory) if
                   os.path.isdir(os.path.join(base_directory, f)) and 'erp' in f]

    files = []

    for directory in directories:
        erp_directory_path = f'{base_directory}/{directory}'
        erp_timestamp = [f for f in os.listdir(erp_directory_path) if re.match(r'erp_timestamps_.*\.csv', f)][0]

        files.append(f'{base_directory}/{directory}/{erp_timestamp}')

    return files


total_times = []

for n in range(20):
    ERPTimestamps = get_timestamps(n + 1)

    # Initialize a list to store time differences
    time_diffs = []

    for i in range(8):
        # Initialize a list to store time differences
        trial_time_diffs = []

        ERPEventsDataFrame = read_csv(ERPTimestamps[i])

        # Assuming 'data' is your DataFrame containing the table data
        # Convert 'datetimes' column to datetime objects
        ERPEventsDataFrame['datetimes'] = pd.to_datetime(ERPEventsDataFrame['datetimes'], format='%Y-%m-%d %H.%M.%S.%f')

        # Filter rows for 'TARGET' and 'KEY PRESS' events
        target_key_press = ERPEventsDataFrame[
            (ERPEventsDataFrame['events'] == 'TARGET') | (ERPEventsDataFrame['events'] == 'KEY PRESS')]

        # Iterate through the filtered data to calculate time differences
        for x in range(len(target_key_press) - 1):
            if target_key_press.iloc[x]['events'] == 'TARGET' and target_key_press.iloc[x + 1]['events'] == 'KEY PRESS':
                time_diff = (target_key_press.iloc[x + 1]['datetimes'] - target_key_press.iloc[x][
                    'datetimes']).total_seconds()
                trial_time_diffs.append(time_diff)

        if len(trial_time_diffs) != 0:
            time_diffs += trial_time_diffs
            avg_time_diff = (sum(trial_time_diffs) / len(trial_time_diffs)) * 1000
            print(f'Average time for Participant #{n + 1} Trial #{i + 1}:', avg_time_diff, "miliseconds")

        else:
            print(f'Average time for Participant #{n + 1} Trial #{i + 1}: UNDEFINED')

    # Calculate the average time difference
    if len(time_diffs) != 0:
        avg_time_diff = (sum(time_diffs) / len(time_diffs)) * 1000
        total_times.append(avg_time_diff)
        print(f'Average time for Participant #{n + 1}:', avg_time_diff, "miliseconds")

    else:
        print(f'Average time for Participant #{n + 1}: UNDEFINED')

avg_total_time_diff = sum(total_times) / len(total_times)

print(f'Average time for all participants', avg_total_time_diff, "miliseconds")