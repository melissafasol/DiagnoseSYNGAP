## have to run package in python 3.6
import os 
import sys
import numpy as np 

from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data 
from idtxl.visualise_graph import plot_network
import matplotlib.pyplot as plt



# a) Generate test data
data = Data()
data.generate_mute_data(n_samples=1000, n_replications=5)
print(data)
