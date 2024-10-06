# Import libraries
import numpy as np # For numerical computations
import pandas as pd # for data manipulation
from obspy import read # Processing Seismological data
from datetime import datetime, timedelta # Date time manipulation
import matplotlib.pyplot as plt # Matlab plotting library
from scipy import signal
import os


# Load catalog
def load_catalog(cat_file):
    try:
        return pd.read_csv(cat_file)
    except FileNotFoundError:
        print(f"File not found: {cat_file}")
        return None

# Filter signal
def filter_signal(st, minfreq=0.1, maxfreq=1.5):
    st_filt = st.copy()
    st_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
    st_filt.filter('lowpass', freq=maxfreq, corners=2)  # Apply a low-pass filter for further noise reduction
    return st_filt

# Plotting function
def plot_filtered_data(tr_times_filt, tr_data_filt, arrival, t, f, sxx):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Time series plot
    ax1.plot(tr_times_filt, tr_data_filt)
    ax1.axvline(x=arrival, color='red', label='Detection')
    ax1.set_xlim([min(tr_times_filt), max(tr_times_filt)])
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_xlabel('Time (s)')

    # Spectrogram plot
    vals = ax2.pcolormesh(t, f, 10 * np.log10(sxx), shading='gouraud')
    ax2.set_xlim([min(tr_times_filt), max(tr_times_filt)])
    ax2.set_xlabel(f'Time (Day Hour:Minute)', fontweight='bold')
    ax2.set_ylabel('Frequency (Hz)', fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(vals, ax=ax2, orientation='horizontal')
    cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')

    plt.show()

# Main processing
cat_directory = '../data/lunar/training/catalogs/'
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv'
cat = load_catalog(cat_file)

if cat is not None:
    data_directory = '../data/lunar/training/data/S12_GradeA/'

    # Loop through each row in the catalog
    for idx, row in cat.iterrows():
        try:
            print(f"Processing event {idx+1}/{len(cat)}...")

            # Parse arrival time
            arrival_time = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], '%Y-%m-%dT%H:%M:%S.%f')
            arrival_time_rel = row['time_rel(sec)']
            test_filename = row.filename

            # Load data
            mseed_file = f'{data_directory}{test_filename}.mseed'
            st = read(mseed_file)  # This may raise a FileNotFoundError

            # Filter signal
            st_filt = filter_signal(st)
            tr_filt = st_filt.traces[0]
            tr_times_filt = tr_filt.times()
            tr_data_filt = tr_filt.data

            # Spectrogram
            f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)

            # Plot data
            startime = tr_filt.stats.starttime.datetime
            arrival = (arrival_time - startime).total_seconds()
            plot_filtered_data(tr_times_filt, tr_data_filt, arrival, t, f, sxx)

        except FileNotFoundError as e:
            print(f"File not found for event {idx+1}: {test_filename}.mseed")
            continue  # Skip to the next event












