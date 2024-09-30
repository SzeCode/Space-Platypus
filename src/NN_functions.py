
import os
from string import printable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scipint

# Initializes the neural network model
def init_model(num_input,num_hidden_layers = 10, num_neurons_per_layer = 100):

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