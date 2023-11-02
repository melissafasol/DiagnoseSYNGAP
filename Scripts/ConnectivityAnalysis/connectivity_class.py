import os
import numpy as np
import pandas as pd 

from mne_features.bivariate import compute_phase_lock_val, compute_max_cross_corr

#add feature to calculate by frequency band?

class ConnectivityClass:
    
    def __init__(self, filtered_data):
        self.filtered_data = filtered_data

    def calculate_max_cross_corr(self, num_epochs):
        cross_corr_ls = []
        error_ls = []
        for i in np.arange(0, num_epochs, 1):
            try:
                one_epoch_1 = compute_max_cross_corr(sfreq = 250.4, data = self.filtered_data[:, i]) 
                cross_corr_ls.append(one_epoch_1)
            except:
                print(' error for index ' + str(i))
                error_ls.append(i)
        
        cross_corr_array = np.array(cross_corr_ls)
        error_array = np.array(error_ls)
        
        return cross_corr_array, error_array
    
    def calculate_phase_lock_value(self, num_epochs):
        phase_lock_ls = []
        error_ls = []
        for i in np.arange(0, num_epochs, 1):
            try:
                one_epoch_1 = compute_phase_lock_val(sfreq = 250.4, data = self.filtered_data[:, i]) 
                phase_lock_ls.append(one_epoch_1)
            except:
                print(' error for index ' + str(i))
                error_ls.append(i)
        
        phase_lock_array = np.array(phase_lock_ls)
        error_array = np.array(error_ls)
        
        return phase_lock_array, error_array
    
    
    