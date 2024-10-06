import os
from string import printable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scipint

##Writte a couple filter functions that will take raw data and filter out noise. 
##The output of those function will be returned as an ouput
##The output from multiple filter functions will be compared based on an evaluation
##criteria and the best output data will be chosen to be used further in the code


