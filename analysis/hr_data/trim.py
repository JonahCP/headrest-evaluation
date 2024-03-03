from datetime import datetime
from glob import glob
from pandas import DataFrame, read_csv
from queue import Queue
from time import time

# Specify the folder of interest to trim data
folder_name = 'Feb27_Jason'

dir = f"./{folder_name}/"

# Define CSV trimming function
def trim_csv(timestamp_path, data_path, output_path):
    # Read in timestamps CSV
    timestamps = read_csv(timestamp_path)
    ev = timestamps['events'].tolist()
    dt = timestamps['datetimes'].tolist()

    # Read in and prune headrest data file
    data = read_csv(data_path, index_col = False)
    data.drop(inplace = True, columns = ['raw', 'delta', 'theta', 'alpha', 'beta1', 'beta2', 'x', 'y', 'z',
                'chnl-1-delta', 'chnl-1-theta', 'chnl-1-alpha', 'chnl-1-beta1', 'chnl-1-beta2',
                'chnl-2-delta', 'chnl-2-theta', 'chnl-2-alpha', 'chnl-2-beta1', 'chnl-2-beta2',
                'chnl-3-delta', 'chnl-3-theta', 'chnl-3-alpha', 'chnl-3-beta1', 'chnl-3-beta2'])

    # Create output dataframe to track wanted entries
    out = DataFrame(columns = data.columns)

    # Retrieve START timestamp
    event = (dt[0], ev[0])
    assert event[1] == 'START'

    # Find point in data where stimulus began
    row = 0
    event_time = datetime.strptime(event[0], '%Y-%m-%d %H.%M.%S.%f')
    while datetime.strptime(data.loc[row]['datetime'], '%Y-%m-%d %H.%M.%S.%f') < event_time:
        row += 1

    # Retrieve END timestamp
    event = (dt[-1], ev[-1])
    assert event[1] == 'END'

    event_time = datetime.strptime(event[0], '%Y-%m-%d %H.%M.%S.%f')
    while datetime.strptime(data.loc[row]['datetime'], '%Y-%m-%d %H.%M.%S.%f') <= event_time:
        out.loc[len(out.index)] = data.loc[row]
        row += 1
    
    # Output relevant data into separate CSV file
    out.to_csv(output_path, index = False)


# Trim all SSVEP folders (1-8 w/ data and timestamps) into one CSV file
# for i in range(1, 2):
#     ssvep_dir = dir + f'ssvep{i}/'
#     timestamp_path = glob(ssvep_dir + 'ssvep_timestamps*.csv')[0]
#     data_path = glob(ssvep_dir + 'log*.csv')[0]
#     output_path = dir + f'ssvep{i}_trim.csv'
#
#     print(f'Trimming ssvep{i}...')
#     start_time = time()
#     trim_csv(timestamp_path, data_path, output_path)
#     print('CSV files trimmed! Elapsed time: %.2f s\n' % (time() - start_time))


# Trim all ERP folders (1-8 w/ data and timestamps) into one CSV file
for i in range(1, 9):
    erp_dir = dir + f'erp{i}/'
    timestamp_path = glob(erp_dir + 'erp_timestamps*.csv')[0]
    data_path = glob(erp_dir + 'log*.csv')[0]
    output_path = dir + f'erp{i}_trim.csv'

    print(f'Trimming erp{i}...')
    start_time = time()
    trim_csv(timestamp_path, data_path, output_path)
    print('CSV files trimmed! Elapsed time: %.2f s\n' % (time() - start_time))