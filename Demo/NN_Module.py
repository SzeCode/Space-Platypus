# An algorithm that contains the neural network 
#   creating the model to predict the arrival time.
# Alg adapted from https://github.com/jbramburger/DataDrivenDynSyst 
#   Learning Dynamics with Neural Networks\Forecast.ipynb

import os
from string import printable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scipint


# To be replaced by Earthquake data
# Lorenz System function definition

def Lorenz(y, t, rho, b):
    
    dydt = [10.0*(y[1]-y[0]), y[0]*(rho - y[2]) - y[1], y[0]*y[1]-8.0/3*y[2]]

    return dydt

#------ Begin Generating Lorenz Data

# Initializations

ti = 0
tfin = 500
N = 10001
t = np.linspace(ti,tfin,N)
dt = t[2]-t[1]
print("dt = ", dt)


# Lorenz parameter #2
rho = 28.0;
x0 = [1,2,3]
sol_rho28 = scipint.odeint(Lorenz,x0, t, args=(rho,0))

M_end = 500;
fig = plt.figure()
plt.plot(t[:M_end],sol_rho28[:M_end,0],'k') 
plt.title('The Lorenz Attractor rho = 28.0', fontsize = 20)
plt.xlabel('$x$', fontsize = 20)
plt.ylabel('$z$', fontsize = 20)

# ------------------------------------------------------------------------------------------

# Finding the transition points and set countdowns
tJump = np.zeros((N,1));
n = 0
for i in range(N-1):
    curr = sol_rho28[i+1,0]
    prev = sol_rho28[i,0]

    if (np.sign(curr) !=  np.sign(prev)):
        tJump[i+1,0] = 0.0 #1
        for j in range(i+1-n):
            if n==0:
                tJump[j+n,0] = i+1-n-j
            else:
                if j != 0:
                    tJump[j+n,0] = i+1-n-j

        n = i+1
    else:
        tJump[i+1,0] = 1.0 #0


fig = plt.figure()
plt.plot(tJump[:M_end],'k') 
plt.title('T jump', fontsize = 20)
plt.xlabel('$n$', fontsize = 20)
plt.ylabel('$t$', fontsize = 20)


#------ End Generating Lorenz and time till jump Data

#------ Begin Forecastiong using neural networks with the Lorenz Data
xnforward = [] #initialize matrix for training data

xnforward.append(sol_rho28[0:500,:])
xnforward.append(tJump[0:500])

input("Press Enter to continue...")

# Initializes the neural network model
def init_model(num_hidden_layers = 10, num_neurons_per_layer = 100):

    model = tf.keras.Sequential()

    # Input is (x,y,z,rho)
    model.add(tf.keras.Input(3))

    for _ in range(num_hidden_layers):
        #adds the number of layer at each _ hidden layers
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
            activation=tf.keras.activations.get('relu'), 
            kernel_initializer='glorot_normal'))


    # Output is (t)
    model.add(tf.keras.layers.Dense(1))

    return model

def compute_loss(model, xnforward):

    loss = 0
    tpred = model(xnforward[0])
    xnp1 = xnforward[1] 

    loss += tf.reduce_mean(tf.square(tpred-xnp1))

    return loss

def get_grad(model, xnforward):

    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with respect to trainable vriables.
        tape.watch(model.trainable_variables)
        loss = compute_loss(model, xnforward)#, tJump, steps)

    g = tape.gradient(loss, model.trainable_variables)
    del tape

    return loss, g

# get neural network model
num_hidden_layers = 10
num_neurons_per_layer = 100
model = init_model(num_hidden_layers,num_neurons_per_layer) 

# Learning rate chosen as decremental steps
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,3000,8000], [1e-2,1e-3,1e-4,1e-5])

optim = tf.keras.optimizers.Adam(learning_rate=lr)


# add time function from the time package
from time import time

steps = 1


@tf.function
def train_step():
    # Compute current loss and gradient w.r.t. parameters.
    loss, grad_theta = get_grad(model, xnforward)

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

tpred = np.zeros((M,1))

for m in range(1,M):
    tpred[m] = model(sol_rho28[m-1:m,:])


print(tJump[0:50])
print(tpred[0:50])

input("Press Enter to continue...")
DataStartPred = 0
DataEndPred = 1000

fig = plt.figure()
plt.plot(1/10*tpred[DataStartPred:DataEndPred,0],'b--o')
plt.plot(sol_rho28[DataStartPred:DataEndPred,0], 'r.')
plt.title('Forecasting tJump with Neural Networks', fontsize = 20)
plt.xlabel('$t$', fontsize = 20)
plt.ylabel('$x$', fontsize = 20)


plt.show(block=True)

input("Press Enter to continue...")
# Save model
#model.save('Lorenz_models/LorenztJumpPred_rho=28Saw')

## Save data as .mat file
#import scipy.io

#Param = [dt, N, num_hidden_layers, num_neurons_per_layer, CompTime]
#scipy.io.savemat('LorenztJumpPred_rho=28Saw.mat', dict(tpred = tpred, ttrue = tJump, FullSOl = sol_rho28, Param = Param, rho = rho, loss = Loss_hist))