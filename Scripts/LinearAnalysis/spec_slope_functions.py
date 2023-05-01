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
        
        frequency_range = [0.2, 48]
        frequency_values = np.arange(0.2, 48.2, 0.2)
        
        fooof_ls = []
        for epoch in power_array:
            fm = FOOOF()
            fm.report(frequency_values, epoch, frequency_range)
            aperiodic_values = fm.aperiodic_params_
            fooof_ls.append(aperiodic_values)
            
        fooof_array = np.array(fooof_ls)
        return fooof_array
        
        
