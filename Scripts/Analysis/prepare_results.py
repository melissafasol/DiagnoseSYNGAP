import os 
import pandas as pd
import numpy as np


def average_power_arrays(power_array):
    channel_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    chan_power_ls = []
    for chan in channel_list:
        chan_power = np.mean([epoch[chan] for epoch in power_array], axis = 0)
        chan_power_ls.append(chan_power)
    
    all_channels = np.vstack(chan_power_ls)
    return all_channels

