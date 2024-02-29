from datetime import datetime
from glob import glob
from pandas import DataFrame, read_csv
from queue import Queue
from time import time

# Specify the folder of interest to combine data and timestamp files
folder_name = 'Feb23_Carson'

dir = f"./{folder_name}/"

# Define CSV combination function
def combine_csvs(timestamp_path, data_path, output_path):
    # Read timestamps into a queue of tuples (datetime, event)
    timestamps = read_csv(timestamp_path)
    ev_ = timestamps['events'].tolist()
    dt_ = timestamps['datetimes'].tolist()
    events = Queue()
    for n in range(len(ev_)):
        events.put((dt_[n], ev_[n]))
    
    # Read in and prune headrest data file
    data = read_csv(data_path, index_col = False)
    data.drop(inplace = True, columns = ['delta', 'theta', 'alpha', 'beta1', 'beta2', 'x', 'y', 'z',
                'chnl-1-delta', 'chnl-1-theta', 'chnl-1-alpha', 'chnl-1-beta1', 'chnl-1-beta2',
                'chnl-2-delta', 'chnl-2-theta', 'chnl-2-alpha', 'chnl-2-beta1', 'chnl-2-beta2',
                'chnl-3-delta', 'chnl-3-theta', 'chnl-3-alpha', 'chnl-3-beta1', 'chnl-3-beta2'])

    # Create output dataframe to track wanted entries
    out = DataFrame(columns = data.columns)

    # Retrieve first event (must be 'START')
    event = events.get()
    assert event[1] == 'START'

    # Find point in data where stimulus began
    row = 0
    event_time = datetime.strptime(event[0], '%Y-%m-%d %H.%M.%S.%f')
    while datetime.strptime(data.loc[row]['datetime'], '%Y-%m-%d %H.%M.%S.%f') < event_time:
        row += 1

    # Insert event into output data set
    out.loc[len(out.index)] = [event[0], '', '', '', '', '', event[1]]

    # Insert remaining events and relevant data entries to output
    while not events.empty():
        event = events.get()
        event_time = event_time = datetime.strptime(event[0], '%Y-%m-%d %H.%M.%S.%f')

        while datetime.strptime(data.loc[row]['datetime'], '%Y-%m-%d %H.%M.%S.%f') < event_time:
            out.loc[len(out.index)] = data.loc[row]
            row += 1

        out.loc[len(out.index)] = [event[0], '', '', '', '', '', event[1]]
    
    # Output relevant data into separate CSV file
    out.to_csv(output_path, index = False)


# Combine all SSVEP folders (1-8 w/ data and timestamps) into one CSV file
for i in range(1, 9):
    ssvep_dir = dir + f'ssvep{i}/'
    timestamp_path = glob(ssvep_dir + 'ssvep_timestamps*.csv')[0]
    data_path = glob(ssvep_dir + 'log*.csv')[0]
    output_path = dir + f'ssvep{i}.csv'

    print(f'Combining ssvep{i}...')
    start_time = time()
    combine_csvs(timestamp_path, data_path, output_path)
    print('CSV files combined! Elapsed time: %.2f s\n' % (time() - start_time))


# Combine all ERP folders (1-8 w/ data and timestamps) into one CSV file
for i in range(1, 9):
    erp_dir = dir + f'erp{i}/'
    timestamp_path = glob(erp_dir + 'erp_timestamps*.csv')[0]
    data_path = glob(erp_dir + 'log*.csv')[0]
    output_path = dir + f'erp{i}.csv'

    print(f'Combining erp{i}...')
    start_time = time()
    combine_csvs(timestamp_path, data_path, output_path)
    print('CSV files combined! Elapsed time: %.2f s\n' % (time() - start_time))