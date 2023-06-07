import os 
import scipy 
import numpy as np
import pandas as pd 

import fooof
from fooof import FOOOF
from fooof.core.io import save_fm, load_json
from fooof.core.reports import save_report_fm

class SpectralSlope:
    
    def __init__(self, dat_array):
        self.dat_array = dat_array


    def power_calc(self):

        power_ls = []
    
        for epoch in self.dat_array:
            freq, power = scipy.signal.welch(epoch, window='hann', fs=250.4, nperseg=1252)
            freq_interest = power[1:241]
            power_ls.append(freq_interest)
    
        power_array = np.array(power_ls)
    
        return power_array
    
    def fooof_analysis(self, power_array):
        
        frequency_range = [0, 48]
        frequency_values = np.arange(0, 48.2, 0.2)
        
        offset_ls = []
        exponent_ls = []
        error_idx = []
        
        for i, epoch in enumerate(power_array):
            try: 
                fm = FOOOF()
                fm.fit(frequency_values, epoch, frequency_range)
                #FOOOF_results = fm.get_reults() 
                #print(FOOOF_results)
                aperiodic_values = fm.aperiodic_params_
                offset_ls.append(aperiodic_values[0])
                exponent_ls.append(aperiodic_values[1])
            except:
                print('error at index ' + str(i))
                error_idx.append(i)
                
            
        offset_array = np.array(offset_ls)
        exponent_array = np.array(exponent_ls)
        error_array = np.array(error_idx)
        
        return offset_array, exponent_array, error_array
        
        
