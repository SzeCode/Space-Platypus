## Code From demo for the SLA/LTA algorithm

import numpy as np # For numerical computations
import pandas as pd # for data manipulation
from obspy import read # Processing Seismological data
from datetime import datetime, timedelta # Date time manipulation
import matplotlib.pyplot as plt # Matlab plotting library
import os
from scipy.signal import find_peaks

def detect_peaks(tr_data, distance_between_peaks=3000, prominence_threshold=0.5e-8,
                 gradual_fall_window=500, gradual_fall_threshold=0.5e-8,
                 sharp_fall_threshold=0.3e-8):

    #    Detects earthquake peaks and noise from the given data.
    #
    #    Parameters:
    #    - tr_data (numpy array): Input data array containing earthquake measurements.
    #    - distance_between_peaks (int): Minimum distance between detected peaks.
    #    - prominence_threshold (float): Minimum prominence of peaks to be considered.
    #    - gradual_fall_window (int): Number of points to average after the peak.
    #    - gradual_fall_threshold (float): Average threshold for detecting gradual #fall peaks.
    #    - sharp_fall_threshold (float): Average threshold for detecting noise peaks.
    #
    #    Returns:
    #    - tuple: A tuple containing two lists:
    #        - List of indices of detected gradual-fall peaks (earthquake peaks).
    #        - List of indices of detected noise peaks.

    # Parameters for peak detection
    # distance_between_peaks = 15000  # Earthquakes take 14000 points to settle back to normal
    # prominence_threshold = 0.5e-8  # Prominence threshold based on amplitude range
    # gradual_fall_window = 1000  # Points over which we check for a gradual fall
    # gradual_fall_threshold = 1.5e-9  # Maximum decrease over the 500 points for a valid gradual fall
    # sharp_fall_threshold = 3e-9  # Threshold for sharp fall to identify noise

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
                # Calculate the average of the current peak and the next 500 values
                peak_val = tr_data[peak]
                print("peak value",peak_val,"\n")
                average_next_val = np.mean(tr_data[peak+gradual_fall_window:peak+gradual_fall_window + 500])
                print("Avg next value 1",tr_data[peak+gradual_fall_window],"\n")
                print("Avg next value 2",tr_data[peak+gradual_fall_window],"\n")

                print("Avg next value",average_next_val,"\n")
                average_value=peak_val-average_next_val
                print("Avg value",average_value,"\n")

                # If the average is less than the gradual fall threshold, it's a valid peak
                if average_value < gradual_fall_threshold:
                    gradual_fall_peaks.append(peak)
                # Check for sharp fall condition based on average
                elif (tr_data[peak] - np.mean(tr_data[peak + 1:peak + 500])) > sharp_fall_threshold:
                    noise_peaks.append(peak)

        return gradual_fall_peaks, noise_peaks

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


    return noise_peaks, noise_peak_heights