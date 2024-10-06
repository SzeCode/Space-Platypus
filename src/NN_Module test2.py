# An algorithm that contains the neural network 
#   creating the model to predict the arrival time.
# Alg adapted from https://github.com/jbramburger/DataDrivenDynSyst 
#   Learning Dynamics with Neural Networks\Forecast.ipynb


import os
from string import printable
from tkinter import END
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
#from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scipint

import pandas as pd # for data manipulation
from obspy import read # Processing Seismological data
from datetime import datetime, timedelta # Date time manipulation

from time import time # add time function from the time package

from NN_functions import init_model_RNN, init_model_Binary, init_model, compute_loss, get_grad,Create_Sliding_Window_Array, Test_Filter, find_Max_Signal
import random


# Directory for the catalogs
cat_directory = '../data/lunar/training/catalogs/' # File directory
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv' # File name 
cat = pd.read_csv(cat_file)


# Get data and arrival times from the catalog
ndata = 5                                                              # Total Number of datasets
MaxV = find_Max_Signal(ndata, cat)                                     # Max velocity across all data
for i in range(0,ndata):

    row = cat.iloc[i]                                                  # from pandas: get 6th row in file 'cat' (iloc = ith location)
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
    
    # tr_data_df = Test_Filter(st, 0.7, 0.3)

    # plt.plot(tr_times_d, tr_data_d)
    # plt.plot(tr_times_d, tr_data_df)
    # plt.show()
    MaxV = max(abs(tr_data_d))
    tr_data_d = (tr_data_d)/(MaxV)                     # Normalize the data between -1 and 1
    

    # Get Relative time of arrival
    startime = tr.stats.starttime.datetime
    arrival = (arrival_time-startime).total_seconds()+90

    # Isolate the large dataset into a smaller manageable signal around the arrival time
    # The arrival time index is at random places in the signal to avoid model biases 
    total_signal_time = 1200                                          # [s]                
    time_before_arrival = random.randrange(200,900)                   # [s]
    time_after_arrival = total_signal_time - time_before_arrival      # [s]
    arrival_index = np.where(tr_times_d >= arrival)[0][0]             # Index of tr_times_d containing the arrival time
    
    start_signal = arrival_index - round(st[0].stats.sampling_rate*time_before_arrival)
    end_signal = arrival_index + round(st[0].stats.sampling_rate*time_after_arrival)

    ## Correct for out of bounds. Still maintains total signal time.
    if start_signal < 0:
        end_signal = end_signal - start_signal
        start_signal = 0
    if end_signal >= len(tr_times_d):
        start_signal = start_signal - (end_signal - (len(tr_times_d) - 1))
        end_signal = len(tr_times_d) - 1

    # Keep same amount of elements
    tr_times = tr_times_d[start_signal:end_signal]
    tr_data = tr_data_d[start_signal:end_signal]

    
    # Sliding window array creation
    Window_Size = 200
    signal_size = len(tr_data)
    nRows, Array_trainingSingleData, Array_desiredwindowOutput, arrival_index = Create_Sliding_Window_Array(Window_Size, signal_size, tr_times, tr_data, arrival, st[0].stats.sampling_rate)
 

    # appending all dataset into one array
    if i == 0:
        Array_training = []                               # Learning data
        desired_output = []                               # True solution
        tr_data_combined = []                             # Isolated Data
        tr_times_combined = []                            # Isolated time
        arrival_combined = []                             # Arrival
        nRows_combined = []                               # Number of windows
        arrivalTindex_combined = []                       # Index in tr_times where arrival occures
        Array_training.append(Array_trainingSingleData)
        desired_output.append(Array_desiredwindowOutput)
        tr_data_combined.append(tr_data)
        tr_times_combined.append(tr_times)
        arrival_combined.append(arrival)
        arrivalTindex_combined.append(arrival_index)
        nRows_combined.append(nRows)
    else:
        Array_training.append(Array_trainingSingleData)
        desired_output.append(Array_desiredwindowOutput)
        tr_data_combined.append(tr_data)
        tr_times_combined.append(tr_times)
        arrival_combined.append(arrival)
        arrivalTindex_combined.append(arrival_index)
        nRows_combined.append(nRows)


input_size = Window_Size 

#------ Creation of neural network model

num_hiddenDense_layers = 20
num_Denseneurons_per_layer = 400
model = init_model_Binary(input_size, num_hiddenDense_layers, num_Denseneurons_per_layer) 


print('model initialized')
model.summary()
input("Press Enter to continue...")


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
#lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

for b in range(0,ndata-1):
    model.fit(Array_training[b], desired_output[b], epochs=20, batch_size=10, validation_data=(Array_training[b+1], desired_output[b+1]))
    print('Array trained', b)


# test_loss = model.evaluate(Array_training[0], desired_output[0])
predictions = model.predict(Array_training[ndata-1])
#predictions1 = model.predict(Array_training[1])

# Post-Processing Predictions
# Window_SizeP = 20
# halfWindow = int(Window_SizeP/2)
# predictionsProcessed = predictions
# for h in range(int(halfWindow),int(len(predictions)-halfWindow)):
#     predictionsProcessed[h] = round(np.mean(predictions[h-halfWindow:h+halfWindow]))


fig,ax = plt.subplots(1,1,figsize=(10,3))
plt.plot(tr_times_combined[ndata-1][0:nRows_combined[0]],tr_data_combined[ndata-1][0:nRows_combined[0]])
plt.plot(tr_times_combined[ndata-1][0:nRows_combined[0]],predictions[:,0])
ax.axvline(x = arrival_combined[ndata-1], color='red',label='Rel.Arrival')


print("\n")
print("Desired: ", arrival_combined[ndata-1])
print("Prediction is: ",tr_times_combined[ndata-1][np.where(np.abs(predictions-1) < 1e-2)[0][0]])


plt.show()
input("Press Enter to continue...")
# Save model

# path = './models'
# if os.path.exists(path):
#     print("Folder %s already exists" % path)
# else: 
#     os.mkdir('./TestOutput')

#model.save('models/Prelim_Model')

## Save data as .mat file
#import scipy.io

#Param = [dt, N, num_hidden_layers, num_neurons_per_layer, CompTime]
#scipy.io.savemat('LorenztJumpPred_rho=28Saw.mat', dict(tpred = tpred, ttrue = tJump, FullSOl = sol_rho28, Param = Param, rho = rho, loss = Loss_hist))




# print("\n")
# print("Desired: ", arrival_combined[1])
# print("Prediction is: ",tr_times_combined[1][np.where(np.abs(predictions1-1) < 1e-2)[0][0]])



# predictions = model.predict(Array_training[0])
# print("\n")
# print("Desired: ", arrival_combined[0])
# print("Prediction is: ",np.where(np.abs(predictions-1) < 1e-2)[0][0])
# print("Prediction is: ",tr_times_combined[0][np.where(np.abs(predictions-1) < 1e-2)[0][0]])

# predictions = model.predict(Array_training[1])
# print("\n")
# print("Desired: ", arrival_combined[1])
# print("Prediction is: ",np.where(np.abs(predictions-1) < 1e-2)[0][0])
# print("Prediction is: ",tr_times_combined[1][np.where(np.abs(predictions-1) < 1e-2)[0][0]])




#num_hiddenRec_layers = 5
#num_Recneurons_per_layer = 64
#model = init_model_RNN(input_size, num_hiddenDense_layers, num_Denseneurons_per_layer, num_hiddenRec_layers, num_Recneurons_per_layer) 

#print(Array_training)
#print(model(Array_training))

# Learning rate chosen as decremental steps
# lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([5000,10000,15000], [0.1,1e-2,1e-3,1e-4])

# optim = tf.keras.optimizers.Adam(learning_rate=lr)

# steps = 1

# @tf.function
# def train_step():
#     # Compute current loss and gradient w.r.t. parameters.
#     loss, grad_theta = get_grad(model, Array_training, desired_output)

#     # Perform gradient descent step
#     optim.apply_gradients(zip(grad_theta, model.trainable_variables))

#     return loss

# # Number of training epochs
# N_training = 20000
# Loss_hist = [] # Matrix to collect losses

# # Start timer
# t0 = time()


# # Train the data
# for i in range(N_training+1):
#     loss = train_step()

#     Loss_hist.append(loss.numpy())

#     if i%50 == 0:
#         print('It {:05d}: loss = {:10.8e}'.format(i,loss))

# # Print overal computation time
# CompTime = time()-t0
# print('\nComputation time:{} seconds'.format(CompTime))



# # Use Trained Model to Forecast
# M = 1100

# #tpred = np.zeros((M,1))

# #for m in range(1,M):
# #    tpred[m] = model(sol_rho28[m-1:m,:])

# Guess = model(Array_training)
# print('predicted: %d',Guess)

#input("Press Enter to continue...")
#DataStartPred = 0
#DataEndPred = 1000

#fig = plt.figure()
#plt.plot(1/10*tpred[DataStartPred:DataEndPred,0],'b--o')
#plt.plot(sol_rho28[DataStartPred:DataEndPred,0], 'r.')
#plt.title('Forecasting tJump with Neural Networks', fontsize = 20)
#plt.xlabel('$t$', fontsize = 20)
#plt.ylabel('$x$', fontsize = 20)


#plt.show(block=True)