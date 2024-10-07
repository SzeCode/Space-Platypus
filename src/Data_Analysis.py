## Code From demo for the SLA/LTA algorithm

import numpy as np # For numerical computations
import pandas as pd # for data manipulation
from obspy import read # Processing Seismological data
from datetime import datetime, timedelta # Date time manipulation
import matplotlib.pyplot as plt # Matlab plotting library
import os
from scipy.signal import find_peaks

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

# Parameters for peak detection
distance_between_peaks = 15000  # Earthquakes take 14000 points to settle back to normal
prominence_threshold = 0.5e-8  # Prominence threshold based on amplitude range
gradual_fall_window = 10000  # Points over which we check for a gradual fall
gradual_fall_threshold = 0.5e-8  # Minimum decrease over the 14000 points for a valid gradual fall
sharp_fall_threshold = 0.3e-8  # Threshold for sharp fall to identify noise

# Step 1: Find all peaks
peaks, properties = find_peaks(tr_data, distance=distance_between_peaks, prominence=prominence_threshold)

# Step 2: Analyze the gradual fall after each peak
gradual_fall_peaks = []
noise_peaks = []
for peak in peaks:
    # Only consider positive peaks
    if tr_data[peak] > 0:
        # Check if we have enough data after the peak to check for a gradual fall
        if peak < len(tr_data) - gradual_fall_window:
            # Check if the amplitude gradually decreases over the defined window (3000 points)
            fall_slope = tr_data[peak] - tr_data[peak + gradual_fall_window]

            # If the amplitude decreases gradually and meets the threshold, it's an earthquake peak
            if fall_slope >= gradual_fall_threshold:  # Ensure the decrease is gradual
                gradual_fall_peaks.append(peak)
            # If the amplitude falls sharply within a few points, mark as noise
            elif tr_data[peak] - tr_data[peak + 5] > sharp_fall_threshold:
                noise_peaks.append(peak)

# Convert gradual_fall_peaks and noise_peaks to NumPy arrays
gradual_fall_peaks = np.array(gradual_fall_peaks, dtype=int)
noise_peaks = np.array(noise_peaks, dtype=int)

# Step 3: Get the heights of the detected gradual-fall peaks
gradual_fall_peak_heights = tr_data[gradual_fall_peaks]
noise_peak_heights = tr_data[noise_peaks]

# Sort the gradual-fall peaks by height in descending order and get the top 5
top_5_indices = np.argsort(gradual_fall_peak_heights)[-5:][::-1]
top_5_peaks = gradual_fall_peaks[top_5_indices]
top_5_heights = gradual_fall_peak_heights[top_5_indices]

# Output the peak indices and heights
print("Top 5 gradual-fall peak indices:", top_5_peaks)
print("Top 5 peak heights:", top_5_heights)
print("Noise peak indices:", noise_peaks)
print("Noise peak heights:", noise_peak_heights)

# Step 4: Plot the data and detected peaks (gradual-fall and noise)
plt.figure(figsize=(10, 6))
plt.plot(tr_data, label='Data')

# Plot gradual-fall peaks in red
plt.plot(top_5_peaks, top_5_heights, "x", label='Top 5 Gradual-Fall Peaks', color='red', markersize=10)

# Plot noise peaks in blue
plt.plot(noise_peaks, noise_peak_heights, "o", label='Noise Peaks', color='blue', markersize=5)

plt.title('Detected Gradual-Fall and Noise Peaks in Earthquake Data')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.legend()
plt.show()