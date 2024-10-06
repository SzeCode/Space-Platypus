# Import libraries
import numpy as np # For numerical computations
import pandas as pd # for data manipulation
from obspy import read # Processing Seismological data
from datetime import datetime, timedelta # Date time manipulation
import matplotlib.pyplot as plt # Matlab plotting library
import os

#cat_directory = 'C:/Users/Alexandr/Documents/GitHub/Space-Platypus/data/lunar/training/catalogs/' # File directory
cat_directory = './data/lunar/training/catalogs/' # File directory
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv' # File name 
cat = pd.read_csv(cat_file)

print(cat)

row = cat.iloc[8] # from pandas: get 6th row in file 'cat' (iloc = ith location)

# Absolute arrival time (start time of seismic event)
arrival_time = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'],'%Y-%m-%dT%H:%M:%S.%f')
print(arrival_time)

# Relative time (sec)
arrival_time_rel = row['time_rel(sec)']
print(arrival_time_rel)

# Filename of 6th row
test_filename = row.filename
print(test_filename)


# Find file containing signal read csv
data_directory = './data/lunar/training/data/S12_GradeA/'
csv_file = f'{data_directory}{test_filename}.csv'

data_cat = pd.read_csv(csv_file)
print(data_cat)


# Read in time steps and velocities

csv_times = np.array(data_cat['time_rel(sec)'].tolist())
csv_data = np.array(data_cat['velocity(m/s)'].tolist())

# plot figures
fig,ax = plt.subplots(1,1,figsize=(10,3))
ax.plot(csv_times,csv_data)

ax.set_xlim([min(csv_times),max(csv_times)])
ax.set_ylabel('Velocity (m/s)')
ax.set_xlabel('Time (s)')
ax.set_title(f'{test_filename}',fontweight='bold')

# Plot red vertical line indicating arrival time
arrival_line = ax.axvline(x=arrival_time_rel, c='red', label='rel. Arrival')
ax.legend(handles=[arrival_line])

#plt.show()

# Another method (Recommended due to speed)
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


fig,ax = plt.subplots(1,1,figsize=(10,3))
ax.plot(tr_times,tr_data)

ax.axvline(x = arrival, color='red',label='Rel.Arrival')
ax.legend(loc='upper left')

ax.set_xlim([min(tr_times),max(tr_times)])
ax.set_ylabel('Velocity (m/s)')
ax.set_xlabel('Time (s)')
ax.set_title(f'{test_filename}',fontweight='bold')

plt.show()


# Filtering Trace using Band pass filter

minfreq = 0.5
maxfreq = 1.0

st_filt = st.copy()
st_filt.filter('bandpass',freqmin=minfreq,freqmax=maxfreq)

tr_filt = st_filt.traces[0].copy()
tr_times_filt = tr_filt.times()
tr_data_filt = tr_filt.data

from scipy import signal
from matplotlib import cm

f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)



fig = plt.figure(figsize=(10,10))

ax = plt.subplot(2,1,1)
ax.plot(tr_times_filt, tr_data_filt)
ax.axvline(x = arrival, color='red', label = 'Detection')

ax.set_xlim([min(tr_times_filt),max(tr_times_filt)])
ax.set_ylabel('Velocity (m/s)')
ax.set_xlabel('Time (s)')

ax2 = plt.subplot(2, 1, 2)
vals = ax2.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
ax2.axvline(x=arrival, c='red')

ax2.set_xlim([min(tr_times_filt),max(tr_times_filt)])
ax2.set_xlabel(f'Time (Day Hour:Minute)', fontweight='bold')
ax2.set_ylabel('Frequency (Hz)', fontweight='bold')

cbar = plt.colorbar(vals, orientation='horizontal')
cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')

#plt.show()


## STA/LTA
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import highpass
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset

df = tr.stats.sampling_rate

sta_len = 120 # seconds
lta_len = 600 # seconds

# Characteristic Function (ratio of amplitudes between short_term and long term)
cft = classic_sta_lta(tr_data, int(sta_len * df), int(lta_len * df))

fig,ax = plt.subplots(1,1,figsize=(12,3))
ax.plot(tr_times,cft)

ax.set_xlim([min(tr_times),max(tr_times)])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Characteristic function')

#plt.show()


# Thresholds on the characteristic function for start and finish quakes
thr_on = 4
thr_off = 1.5

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


## Exporting into a catalog

fname = row.filename
starttime = tr.stats.starttime.datetime

detection_times = []
fnames = []

for i in np.arange(0, len(on_off)):
    triggers = on_off[i]
    on_time = starttime + timedelta(seconds = tr_times[triggers[0]])
    on_time_str = datetime.strftime(on_time,'%Y-%m-%dT%H:%M:%S.%f')
    detection_times.append(on_time_str)
    fnames.append(fname)

detect_df = pd.DataFrame(data = {'filename':fnames, 'time_abs(%Y-%m-%dT%H:%M:%S.%f)':detection_times, 'time_rel(sec)':tr_times[triggers[0]]})
print(detect_df.head())

path = './TestOutput'
if os.path.exists(path):
    print("Folder %s already exists" % path)
else: 
    os.mkdir('./TestOutput')

detect_df.to_csv(path + '/catalog.csv', index=False)
print("File saved")
