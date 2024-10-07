# Import libraries
import numpy as np # For numerical computations
import pandas as pd # for data manipulation
from obspy import read # Processing Seismological data
from datetime import datetime, timedelta # Date time manipulation
import matplotlib.pyplot as plt # Matlab plotting library
from scipy import signal
import os

#Old Version 

cat_directory = './data/lunar/training/catalogs/' # File directory
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv' # File name 
try:   #Error Handling                      
    cat = pd.read_csv(cat_file)
except FileNotFoundError:
    print(f"File not found: {cat_file}")


print(cat)

row = cat.iloc[6] # from pandas: get 6th row in file 'cat' (iloc = ith location)

# Absolute arrival time (start time of seismic event)
arrival_time = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'],'%Y-%m-%dT%H:%M:%S.%f')
print(arrival_time)

# Relative time (sec)
arrival_time_rel = row['time_rel(sec)']
print(arrival_time_rel)

# Filename of 6th row
test_filename = row.filename
print(test_filename)


# Another method (Recommended due to speed)
# Find file containing signal read miniseed (time series data)
data_directory = './data/lunar/training/data/S12_GradeA/'
mseed_file = f'{data_directory}{test_filename}.mseed'
st = read(mseed_file)





#Filtering

# Adjust the frequency range for better noise reduction
minfreq = 0.1  # Lower cutoff to capture the quake signal
maxfreq = 1.5  # Upper cutoff to reduce noise

# Copy and apply the bandpass filter
st_filt = st.copy()
# getting data and time
tr = st_filt.traces[0].copy()
tr_times = tr.times()
tr_data = tr.data

# Relative times
startime = tr.stats.starttime.datetime
arrival = (arrival_time-startime).total_seconds()
st_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)

# Apply a low-pass filter for further noise reduction
st_filt.filter('lowpass', freq=maxfreq, corners=2)

# Extract the filtered trace and times
tr_filt = st_filt.traces[0].copy()
tr_times_filt = tr_filt.times()
tr_data_filt = tr_filt.data

# Generate the spectrogram
f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot the filtered trace with arrival detection
ax1.plot(tr_times_filt, tr_data_filt)
ax1.axvline(x=arrival, color='red', label='Detection')
ax1.set_xlim([min(tr_times_filt), max(tr_times_filt)])
ax1.set_ylabel('Velocity (m/s)')
ax1.set_xlabel('Time (s)')

# Plot the spectrogram
vals = ax2.pcolormesh(t, f, 10 * np.log10(sxx), shading='gouraud')
ax2.set_xlim([min(tr_times_filt), max(tr_times_filt)])
ax2.set_xlabel(f'Time (Day Hour:Minute)', fontweight='bold')
ax2.set_ylabel('Frequency (Hz)', fontweight='bold')

# Add colorbar
cbar = plt.colorbar(vals, ax=ax2, orientation='horizontal')
cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')

plt.show()
