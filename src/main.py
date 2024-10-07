import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from datetime import datetime, timedelta  # Date time manipulation
from time import time  # add time function from the time package

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # for data manipulation
import scipy.integrate as scipint
import tensorflow as tf
from keras.utils import plot_model
from obspy import read  # Processing Seismological data

from Data_Analysis_using_avg_fn import detect_peaks
from NN_functions import reformInputData

Data_FileNum = 11


model = tf.keras.models.load_model("models/Model4_Arrival_Time")
print(model.summary())
# plot_model(model, to_file="keras_model_chart.png")

# Directory for the catalogs
cat_directory = "../data/lunar/training/catalogs/"  # File directory
cat_file = cat_directory + "apollo12_catalog_GradeA_final.csv"  # File name
cat = pd.read_csv(cat_file)

row = cat.iloc[
    Data_FileNum
]  # from pandas: get 6th row in file 'cat' (iloc = ith location)
arrival_time = datetime.strptime(  # Absolute arrival time (start time of seismic event)
    row["time_abs(%Y-%m-%dT%H:%M:%S.%f)"], "%Y-%m-%dT%H:%M:%S.%f"
)
arrival_time_rel = row["time_rel(sec)"]  # Relative time (sec)

test_filename = row.filename  # Filename

# Find file containing signal read mseed
data_directory = "../data/lunar/training/data/S12_GradeA/"
mseed_file = f"{data_directory}{test_filename}.mseed"
st = read(mseed_file)

# Get data and time from mseed file
tr = st.traces[0].copy()
tr_times_d = tr.times()  # time              [s]
tr_data_d = tr.data  # signal (velocity) [m/s]


# Get Relative time of arrival
startime = tr.stats.starttime.datetime
arrival = (arrival_time - startime).total_seconds()

# -------------------------------------------------------------------------------------------------

top_5_peaks, top_5_heights = detect_peaks(
    tr_data_d,
    distance_between_peaks=3000,
    prominence_threshold=0.5e-8,
    gradual_fall_window=500,
    gradual_fall_threshold=0.5e-8,
    sharp_fall_threshold=0.3e-8,
)


# To replace by estimated peaks STALTA
Est_Arrival = tr_times_d[top_5_heights[0]]
nRows, ArraySlidingWindow, tr_times, tr_data = reformInputData(
    st, tr_times_d, tr_data_d, Est_Arrival, 1800
)


predictions = model.predict(ArraySlidingWindow)
print(f"predictions: {predictions}")


# # Post-Processing Predictions
# Window_SizeP = 20
# halfWindow = int(Window_SizeP/2)
# predictionsProcessed = predictions.copy()
# for h in range(int(halfWindow),int(len(predictions)-halfWindow)):
#     predictionsProcessed[h,0] = round(np.mean(predictions[h-halfWindow:h+halfWindow,0]))


# fig, ax = plt.subplots(1, 1, figsize=(10, 3))
# plt.plot(tr_times[0:nRows], tr_data[0:nRows])
# plt.plot(tr_times[0:nRows], predictions[:, 0])
# ax.axvline(x=arrival, color="red", label="Rel.Arrival")
# ax.axvline(
#     x=tr_times[np.where(np.abs(predictions) < 1e-1)[0][-1]],
#     color="green",
#     label="Rel.Arrival",
# )


print("\n")
print("Desired: ", arrival)
print("Prediction is: ", tr_times[np.where(np.abs(predictions) < 1e-1)[0][-1]])

nn_prediction = tr_times[np.where(np.abs(predictions) < 1e-1)[0][-1]]

plt.show()


plt.close()
plt.plot(tr_times_d, tr_data_d)
plt.axvline(x=nn_prediction, label="Neural Network prediction", color="red")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Predicted Moonquake Arrival Time")
plt.show()

input("Press Enter to continue...")


def export_to_catalog(filename: str, detection_time_rel: float, start_time) -> None:
    ## Exporting into a catalog

    # fname = row.filename
    # starttime = tr.stats.starttime.datetime

    fnames = [filename]

    on_time = start_time + timedelta(seconds=detection_time_rel)

    # print(f"on_time: {on_time}")
    # print(f"start_time: {start_time}")

    on_time_str = datetime.strftime(on_time, "%Y-%m-%dT%H:%M:%S.%f")
    detection_times = [on_time_str]

    # for i in np.arange(0, len(on_off)):
    #     triggers = on_off[i]
    #     on_time = starttime + timedelta(seconds=tr_times[triggers[0]])
    #     on_time_str = datetime.strftime(on_time, "%Y-%m-%dT%H:%M:%S.%f")
    #     detection_times.append(on_time_str)
    #     fnames.append(fname)

    detect_df = pd.DataFrame(
        data={
            "filename": fnames,
            "time_abs(%Y-%m-%dT%H:%M:%S.%f)": detection_times,
            "time_rel(sec)": [detection_time_rel],
        }
    )
    print(detect_df.head())

    # path = "./TestOutput"
    # if os.path.exists(path):
    #     print("Folder %s already exists" % path)
    # else:
    #     os.mkdir("./TestOutput")

    detect_df.to_csv("catalog.csv", index=False)
    print("File saved")


export_to_catalog(
    filename=test_filename,
    detection_time_rel=nn_prediction,
    start_time=tr.stats.starttime.datetime,
)
