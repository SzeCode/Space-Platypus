
import os
from string import printable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scipint
import obspy.signal
from datetime import datetime, timedelta # Date time manipulation
from obspy import read # Processing Seismological data

def find_Max_Signal(ndata, cat):
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
        tr_data_d = tr.data     # signal
        DataSize[i] = max(abs(tr_data_d))

    MaxVel = max(DataSize)    
        
    return MaxVel

def Test_Filter(st_filt, a, b):

    minfreq = 0.5
    maxfreq = 1.0

    #st_filt = st.copy()
    tr_filt = st_filt.traces[0].copy()
    tr_times = tr_filt.times()
    tr_data = tr_filt.data
    
    #.filter('bandpass',freqmin=minfreq,freqmax=maxfreq,corners=1, zerophase=True)
    tr_data_filt = np.zeros((len(tr_data),1))
    tr_data_filt[0,0] = tr_data[0]
    for i in range(0,len(tr_data)-1):
        tr_data_filt[i+1,0] = tr_data_filt[i,0]*a + b*tr_data[i]

    return tr_data_filt
    

def Create_Sliding_Window_Array(Window_Size, signal_size, tr_times, tr_data, arrival, Fs):

    nRows = int(signal_size-Window_Size)
    Array_trainingSingleData = np.zeros((nRows,Window_Size))
    Array_desiredwindowOutput = np.zeros((nRows,1))
    k = 0
    arrival_index = np.where(tr_times >= arrival)[0][0]
    while k < nRows:

        inputNode = np.zeros((1,Window_Size))
        inputNodet = np.zeros((1,Window_Size))

        if k+Window_Size >= signal_size:
            signal_size - k   # = 400
            inputNode[0,0:signal_size - k] = tr_data[k:signal_size]
            inputNodet[0,0:signal_size - k] = tr_data[k:signal_size]
            inputNode[0,signal_size - k+1:Window_Size] = 0
            inputNodet[0,signal_size - k+1:Window_Size] = 0
        else:
            inputNode[0,:] = tr_data[k:k+Window_Size]
            inputNodet[0,:] = tr_times[k:k+Window_Size]

        Array_trainingSingleData[k,:] = inputNode

        if arrival_index <= k:
            Array_desiredwindowOutput[k,0] = 1
        else:
            Array_desiredwindowOutput[k,0] = 0

        k = k + 1
    
    fig,ax = plt.subplots(1,1,figsize=(10,3))
    plt.plot(tr_times[0:signal_size-Window_Size],tr_data[0:signal_size-Window_Size])
    plt.plot(tr_times[0:signal_size-Window_Size],Array_desiredwindowOutput[:,0])
    #ax.plot(tr_times,tr_data)
    ax.axvline(x = arrival, color='red',label='Rel.Arrival')
    ax.axvline(x = arrival+Window_Size/Fs, color='blue',label='Rel.Arrival')
    plt.show()

    # size [batch_size, sequence_length, feature_size] required    
    Array_trainingSingleData = Array_trainingSingleData.reshape((Array_trainingSingleData.shape[0], Array_trainingSingleData.shape[1], 1))  # Reshape to [batch_size, sequence_length, feature_size]
    Array_desiredwindowOutput = Array_desiredwindowOutput.reshape((Array_trainingSingleData.shape[0], 1))    


    return nRows, Array_trainingSingleData, Array_desiredwindowOutput, arrival_index




# Initializes the neural network model
def init_model(num_input, num_hidden_layers = 10, num_neurons_per_layer = 100):

    model = tf.keras.Sequential()

    # Input is (x,y,z,rho)
    model.add(tf.keras.Input(num_input))



    for _ in range(num_hidden_layers):
        #adds the number of layer at each _ hidden layers
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
            activation=tf.keras.activations.get('relu'), 
            kernel_initializer='glorot_normal'))


    # Output is (t)
    model.add(tf.keras.layers.Dense(1))

    return model

def init_model_Binary(num_input, num_hidden_layers = 10, num_neurons_per_layer = 100):

    model = tf.keras.Sequential()

    # Input elements
    model.add(tf.keras.Input(num_input))
    
    #model.add(tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(num_input, 1)))
    #model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    #model.add(tf.keras.layers.LSTM(32, return_sequences=False))

    for _ in range(num_hidden_layers):
        #adds the number of layer at each _ hidden layers
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
            activation=tf.keras.activations.get('relu'), 
            kernel_initializer='glorot_normal'))

    model.add(tf.keras.layers.Dense(128,
            activation=tf.keras.activations.get('relu'), 
            kernel_initializer='glorot_normal'))
    model.add(tf.keras.layers.Dense(64,
        activation=tf.keras.activations.get('relu'), 
        kernel_initializer='glorot_normal'))
    model.add(tf.keras.layers.Dense(32,
        activation=tf.keras.activations.get('relu'), 
        kernel_initializer='glorot_normal'))

    # Output
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

    return model

# Initializes the neural network model
def init_model_RNN(num_input, num_hiddenDense_layers = 10, num_Denseneurons_per_layer = 100, num_hiddenRec_layers = 10, num_Recneurons_per_layer = 50):

    model = tf.keras.Sequential()

    # Input is (x,y,z,rho)
    model.add(tf.keras.layers.LSTM(units=100, input_shape=(num_input,1), return_sequences = True))
    

    for _ in range(num_hiddenRec_layers):
        #adds the number of layer at each _ hidden layers
        model.add(tf.keras.layers.LSTM(units=num_Recneurons_per_layer, return_sequences = True))
    
    model.add(tf.keras.layers.LSTM(units=num_Recneurons_per_layer, return_sequences = False))

    #model.add(tf.keras.layers.GlobalAveragePooling1D())

    for _ in range(num_hiddenDense_layers):
        #adds the number of layer at each _ hidden layers
        model.add(tf.keras.layers.Dense(num_Denseneurons_per_layer,
            activation=tf.keras.activations.get('relu'), 
            kernel_initializer='glorot_normal'))

    model.add(tf.keras.layers.Dense(16,
            activation=tf.keras.activations.get('relu'), 
            kernel_initializer='glorot_normal'))
    
    # Output is (t)
    model.add(tf.keras.layers.Dense(1))

    return model

def compute_loss(model, data_input, desired_output):

    loss = 0
    tpred = model(data_input)
    xnp1 = desired_output 

    loss += tf.reduce_mean(tf.square(tpred-xnp1))

    return loss

def get_grad(model, data_input, desired_output):

    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with respect to trainable vriables.
        tape.watch(model.trainable_variables)
        loss = compute_loss(model, data_input, desired_output)#, tJump, steps)

    g = tape.gradient(loss, model.trainable_variables)
    del tape

    return loss, g

#@tf.function
#def train_step(model, xnforward, optim):
#    # Compute current loss and gradient w.r.t. parameters.
#    loss, grad_theta = get_grad(model, xnforward)

#    # Perform gradient descent step
#    optim.apply_gradients(zip(grad_theta, model.trainable_variables))

#    return loss, model, xnforward, optim