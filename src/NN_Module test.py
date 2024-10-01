# An algorithm that contains the neural network 
#   creating the model to predict the arrival time.
# Alg adapted from https://github.com/jbramburger/DataDrivenDynSyst 
#   Learning Dynamics with Neural Networks\Forecast.ipynb

## TO DO: If very large loss persist, try having 2 inputs (time and signal) with 
#  one output (1 or 0) for arrival true or false

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
    print(np.shape(tr_data))

    #print(round(st[0].stats.sampling_rate*101))
    print(tr_times[-1])
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

    plt.show()

    # initialize matrices for training data
    if i == 0:
        Array_training = np.zeros((ndata,np.shape(tr_data)[0])) # Learning data
        desired_output = np.zeros((ndata,1))                    # Desired outcome
    else:
        Array_training[i,:] = tr_data
        desired_output[i,:] = arrival


#print(Array_training)

input_size = np.shape(Array_training[1])[0]

#print(input_size)
#print(desired_output)

input("Press Enter to continue...")


#------ Creation of neural network model

num_hidden_layers = 15
num_neurons_per_layer = 100
model = init_model(input_size,num_hidden_layers,num_neurons_per_layer) 

#print(Array_training)
#print(model(Array_training))

# Learning rate chosen as decremental steps
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([15000,160000,17000], [0.1,1e-2,1e-3,1e-4])

optim = tf.keras.optimizers.Adam(learning_rate=lr)

steps = 1

@tf.function
def train_step():
    # Compute current loss and gradient w.r.t. parameters.
    loss, grad_theta = get_grad(model, Array_training, desired_output)

    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta, model.trainable_variables))

    return loss

# Number of training epochs
N_training = 20000
Loss_hist = [] # Matrix to collect losses

# Start timer
t0 = time()


# Train the data
for i in range(N_training+1):
    loss = train_step()

    Loss_hist.append(loss.numpy())

    if i%50 == 0:
        print('It {:05d}: loss = {:10.8e}'.format(i,loss))

# Print overal computation time
CompTime = time()-t0
print('\nComputation time:{} seconds'.format(CompTime))



# Use Trained Model to Forecast
M = 1100

#tpred = np.zeros((M,1))

#for m in range(1,M):
#    tpred[m] = model(sol_rho28[m-1:m,:])

Guess = model(Array_training)
print('predicted: %d',Guess)

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
