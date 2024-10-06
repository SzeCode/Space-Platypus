## Code From demo for the SLA/LTA algorithm

import numpy as np # For numerical computations
import pandas as pd # for data manipulation
from obspy import read # Processing Seismological data
from datetime import datetime, timedelta # Date time manipulation
import matplotlib.pyplot as plt # Matlab plotting library
import os

cat_directory = './data/lunar/training/catalogs/' # File directory
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv' # File name 
cat = pd.read_csv(cat_file)

print(cat)

row = cat.iloc[1] # from pandas: get 6th row in file 'cat' (iloc = ith location)

# Absolute arrival time (start time of seismic event)
arrival_time = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'],'%Y-%m-%dT%H:%M:%S.%f')
print(arrival_time)

# Relative time (sec)
arrival_time_rel = row['time_rel(sec)']
print(arrival_time_rel)

# Filename of 6th row
test_filename = row.filename
print(test_filename)

# Find file containing signal read miniseed (time series data)
data_directory = './data/lunar/training/data/S12_GradeA/'
mseed_file = f'{data_directory}{test_filename}.mseed'
st = read(mseed_file)
print(st)

print(st[0].stats)

# getting data and time
tr = st.traces[0].copy()
tr_times = tr.times()
tr_data = tr.data

# Relative times
startime = tr.stats.starttime.datetime
arrival = (arrival_time-startime).total_seconds()
print(arrival)

from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import highpass
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset

df = tr.stats.sampling_rate

sta_len = 300 # seconds
lta_len = 10000 # seconds

# Characteristic Function (ratio of amplitudes between short_term and long term)
cft = classic_sta_lta(tr_data, int(sta_len * df), int(lta_len * df))

fig,ax = plt.subplots(1,1,figsize=(12,3))
ax.plot(tr_times,cft)

ax.set_xlim([min(tr_times),max(tr_times)])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Characteristic function')



# Thresholds on the characteristic function for start and finish quakes
thr_on = 10
thr_off = 0.5

# Array of indices with trigger on and trigger off 
on_off = np.array(trigger_onset(cft, thr_on, thr_off))


fig,ax = plt.subplots(1,1,figsize=(12,3))

## For every ith row, show with line
for i in np.arange(0, len(on_off)):
    triggers = on_off[i]
    ax.axvline( x = tr_times[triggers[0]], color='red', label='Trig. On')
    ax.axvline(x = tr_times[triggers[1]], color='purple', label='Trig. Off')


ax.plot(tr_times,tr_data)
ax.set_xlim([min(tr_times),max(tr_times)])
ax.legend()

plt.show()