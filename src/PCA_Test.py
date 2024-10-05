import os
from string import printable
from tkinter import END
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scipint

import pandas as pd # for data manipulation
from obspy import read # Processing Seismological data
from datetime import datetime, timedelta # Date time manipulation

from time import time # add time function from the time package

from NN_functions import init_model
from NN_functions import compute_loss
from NN_functions import get_grad


# Directory for the catalogs
cat_directory = '../data/lunar/training/catalogs/' # File directory
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv' # File name 
cat = pd.read_csv(cat_file)


# Get data and arrival times from each of the rows in the catalog
ndata = 10
for i in range(0,ndata):
    
    row = cat.iloc[i] # from pandas: get 6th row in file 'cat' (iloc = ith location)

    # Absolute arrival time (start time of seismic event)
    arrival_time = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'],'%Y-%m-%dT%H:%M:%S.%f')

    # Relative time (sec)
    arrival_time_rel = row['time_rel(sec)']

    # Filename of 6th row
    test_filename = row.filename

    # Find file containing signal read mseed
    data_directory = '../data/lunar/training/data/S12_GradeA/'
    mseed_file = f'{data_directory}{test_filename}.mseed'
    st = read(mseed_file)

    # Get data and time
    tr = st.traces[0].copy()
    tr_times_d = tr.times() # t
    tr_data_d = tr.data     # signal

    # Get Relative time of arrival
    startime = tr.stats.starttime.datetime
    arrival = (arrival_time-startime).total_seconds()
    print(arrival)

    # Get range of data to learn. 
    time_before_arrival = 1000 # [s]
    time_after_arrival = 1000  # [s]
    arrival_index = np.where(tr_times_d >= arrival)[0][0]
    
    start_signal = arrival_index - round(st[0].stats.sampling_rate*time_before_arrival)
    end_signal = arrival_index + round(st[0].stats.sampling_rate*time_after_arrival)

    ## Correct for out of bounds
    if start_signal < 0:
        end_signal = end_signal - start_signal
        start_signal = 0
    if end_signal >= len(tr_times_d):
        start_signal = start_signal - (end_signal - (len(tr_times_d) - 1))
        end_signal = len(tr_times_d) - 1


    # Keep same amount of elements
    tr_times = tr_times_d[start_signal:end_signal]
    tr_data = tr_data_d[start_signal:end_signal]

    ## FOR DEBUGGING PURPOSES

    #print(len(tr_times))
    #print(tr_data)
    #print(np.shape(tr_data))

    #print(round(st[0].stats.sampling_rate*101))
    #print(tr_times[-1])
    #print(np.where(tr_times >= arrival)[0][0])
    #print(tr_times[np.where(tr_times >= arrival)[0][0]])
    

    #fig,ax = plt.subplots(1,1,figsize=(10,3))
    #ax.plot(tr_times,tr_data)

    #ax.axvline(x = arrival, color='red',label='Rel.Arrival')
    #ax.legend(loc='upper left')

    #ax.set_xlim([min(tr_times),max(tr_times)])
    #ax.set_ylabel('Velocity (m/s)')
    #ax.set_xlabel('Time (s)')
    #ax.set_title(f'{test_filename}',fontweight='bold')

    #plt.show()

    # initialize matrices for training data
    if i == 0:
        Array_training = np.zeros((ndata,np.shape(tr_data)[0])) # Learning data
        desired_output = np.zeros((ndata,1))                    # Desired outcome
    else:
        arrival_index = np.where(tr_times >= arrival)[0][0]
        Array_training[i,:] = tr_data
        desired_output[i,:] = arrival_index
        print(arrival_index)