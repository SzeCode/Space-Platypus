# An algorithm that contains the neural network 
#   creating the model to predict the arrival time.
# Alg adapted from https://github.com/jbramburger/DataDrivenDynSyst 
#   Learning Dynamics with Neural Networks\Forecast.ipynb

## TO DO: Sliding window approach

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

from NN_functions import init_model_RNN
from NN_functions import compute_loss
from NN_functions import get_grad


# Directory for the catalogs
cat_directory = '../data/lunar/training/catalogs/' # File directory
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv' # File name 
cat = pd.read_csv(cat_file)


# Get data and arrival times from each of the rows in the catalog
ndata = 3
DataSize = np.zeros(ndata)
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
    DataSize[i] = len(tr_data_d)
    
min_Data_size = min(DataSize)


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

    tr_data_d = (tr_data_d-min(tr_data_d))/(max(tr_data_d) - min(tr_data_d))
    #plt.plot(tr_times_d,tr_data_d)
    #plt.show()

    # Get Relative time of arrival
    startime = tr.stats.starttime.datetime
    arrival = (arrival_time-startime).total_seconds()
    #print(arrival)

    time_before_arrival = 600 # [s]
    time_after_arrival = 600  # [s]
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
    tr_times = tr_times_d#[start_signal:end_signal]
    tr_data = tr_data_d#[start_signal:end_signal]


    Window_Size = 1000
    
    signal_size = len(tr_data)
    nRows = int(signal_size//Window_Size+1)
    #nRows = int(signal_size-Window_Size)
    Array_trainingSingleData = np.zeros((nRows,Window_Size))
    Array_desiredwindowOutput = np.zeros((nRows,1))
    k = 0
    Wk = 0
    p = 0
    arrival_index = np.where(tr_times >= arrival)[0][0]
    #X_train = np.random.randn(1000, 10, 1)
    #y_train = np.random.randn(1000, 1)
    while p < nRows:
        Wk = k
        inputNode = np.zeros((1,Window_Size))
        inputNodet = np.zeros((1,Window_Size))
        for j in range(0,Window_Size):
            
            if k >= signal_size:
                inputNode[0,j] = 0
                inputNodet[0,j] = 0
            else:
                inputNode[0,j] = tr_data[k]
                inputNodet[0,j] = tr_times[k]

            k = k + 1

        Array_trainingSingleData[p,:] = inputNode

        if Wk <= arrival_index and arrival_index < k:
            Array_desiredwindowOutput[p,0] = 1
        else:
            Array_desiredwindowOutput[p,0] = -1
        
        p = p+1

        

    # plt.figure()
    # #plt.plot(tr_times[0:signal_size-Window_Size],tr_data[0:signal_size-Window_Size])
    # plt.plot(Array_desiredwindowOutput[:,0])
    # plt.show()

    # size [batch_size, sequence_length, feature_size] required    
    Array_trainingSingleData = Array_trainingSingleData.reshape((Array_trainingSingleData.shape[0], Array_trainingSingleData.shape[1], 1))  # Reshape to [batch_size, sequence_length, feature_size]
    Array_desiredwindowOutput = Array_desiredwindowOutput.reshape((Array_trainingSingleData.shape[0], 1))    
    #print(np.shape(Array_trainingRow))
    
    #

    # Get range of data to learn. 
    

    ## FOR DEBUGGING PURPOSES

  

    # initialize matrices for training data
    if i == 0:
        Array_training = [] #np.zeros((ndata,nRows)) # Learning data
        desired_output = [] #np.zeros((ndata,1))                    # Desired outcome
        tr_combined = []
        arrival_combined = []
        Array_training.append(Array_trainingSingleData)
        desired_output.append(Array_desiredwindowOutput)
        tr_combined.append(tr_data)
        arrival_combined.append(arrival)
    else:
        # TO DO: change to np.concatenate((X1,X2), axis = 0)
        Array_training.append(Array_trainingSingleData)
        desired_output.append(Array_desiredwindowOutput)
        tr_combined.append(tr_data)
        arrival_combined.append(arrival)


#print(Array_training)

input_size = Window_Size #np.shape(Array_training[1])[0]

#print(input_size)
#print(desired_output)

input("Press Enter to continue...")


#------ Creation of neural network model

num_hiddenDense_layers = 10
num_Denseneurons_per_layer = 64
num_hiddenRec_layers = 2
num_Recneurons_per_layer = 64
model = init_model_RNN(input_size, num_hiddenDense_layers, num_Denseneurons_per_layer, num_hiddenRec_layers, num_Recneurons_per_layer) 

print('model initialized')


model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(Array_training[0], desired_output[0], epochs=20, batch_size=32)

# test_loss = model.evaluate(Array_training[0], desired_output[0])
predictions = model.predict(Array_training[0])

print("\n")
print("Desired: ", arrival_combined[0])
print("Predictions are: ",tr_combined[np.where(np.abs(predictions-1) < 1e-2)[0][-1]][0])



model.fit(Array_training[1], desired_output[1], epochs=20)

predictions = model.predict(Array_training[0])

print("\n")
print("Desired: ", arrival_combined[0])
print("Predictions are: ",tr_combined[np.where(predictions == 1)[-1][0]][0])








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

input("Press Enter to continue...")
# Save model
#model.save('Lorenz_models/LorenztJumpPred_rho=28Saw')

## Save data as .mat file
#import scipy.io

#Param = [dt, N, num_hidden_layers, num_neurons_per_layer, CompTime]
#scipy.io.savemat('LorenztJumpPred_rho=28Saw.mat', dict(tpred = tpred, ttrue = tJump, FullSOl = sol_rho28, Param = Param, rho = rho, loss = Loss_hist))
