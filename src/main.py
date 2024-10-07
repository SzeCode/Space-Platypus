import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scipint
import pandas as pd # for data manipulation
from obspy import read # Processing Seismological data
from datetime import datetime, timedelta # Date time manipulation
from time import time # add time function from the time package
from NN_functions import reformInputData


Data_FileNum = 7


model = tf.keras.models.load_model('models/Model4_Arrival_Time')

# Directory for the catalogs
cat_directory = '../data/lunar/training/catalogs/' # File directory
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv' # File name 
cat = pd.read_csv(cat_file)

row = cat.iloc[Data_FileNum]                                                  # from pandas: get 6th row in file 'cat' (iloc = ith location)
arrival_time = datetime.strptime(                                  # Absolute arrival time (start time of seismic event)
    row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'],'%Y-%m-%dT%H:%M:%S.%f')
arrival_time_rel = row['time_rel(sec)']                            # Relative time (sec)

test_filename = row.filename                                       # Filename

# Find file containing signal read mseed
data_directory = '../data/lunar/training/data/S12_GradeA/'
mseed_file = f'{data_directory}{test_filename}.mseed'
st = read(mseed_file)

# Get data and time from mseed file
tr = st.traces[0].copy()
tr_times_d = tr.times()                                           # time              [s]
tr_data_d = tr.data                                               # signal (velocity) [m/s]


# Get Relative time of arrival
startime = tr.stats.starttime.datetime
arrival = (arrival_time-startime).total_seconds()

# To replace by estimated peaks STALTA
Est_Arrival = arrival
nRows, ArraySlidingWindow, tr_times, tr_data = reformInputData(st, tr_times_d, tr_data_d, Est_Arrival, 1200)


predictions = model.predict(ArraySlidingWindow)


# # Post-Processing Predictions
# Window_SizeP = 20
# halfWindow = int(Window_SizeP/2)
# predictionsProcessed = predictions.copy()
# for h in range(int(halfWindow),int(len(predictions)-halfWindow)):
#     predictionsProcessed[h,0] = round(np.mean(predictions[h-halfWindow:h+halfWindow,0]))


fig,ax = plt.subplots(1,1,figsize=(10,3))
plt.plot(tr_times[0:nRows],tr_data[0:nRows])
plt.plot(tr_times[0:nRows],predictions[:,0])
ax.axvline(x = arrival, color='red',label='Rel.Arrival')
ax.axvline(x = tr_times[np.where(np.abs(predictions) < 1e-1)[0][-1]], color='green',label='Rel.Arrival')


print("\n")
print("Desired: ", arrival)
print("Prediction is: ",tr_times[np.where(np.abs(predictions) < 1e-1)[0][-1]])

plt.show()

input("Press Enter to continue...")