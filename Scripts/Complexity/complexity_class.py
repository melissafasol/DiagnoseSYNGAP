import os 
import sys
import numpy as np 
import pandas as pd 
import scipy 
import matplotlib
import mne 

from mne_features.univariate import compute_higuchi_fd, compute_hurst_exp


class ComplexClass:
    
    def __init__(self, complexity_calculation, motor_indices, visual_indices, somatosensory_indices):
        self.complexity_calculation = complexity_calculation
        self.motor_indices = motor_indices
        self.visual_indices = visual_indices
        self.somatosensory_indices = somatosensory_indices
    
    def two_recordings(self, bandpass_filtered_data_1, bandpass_filtered_data_2, region):
        
        region_ls_1 = []
        region_ls_2 = []
        
        if region == 'motor':
            index_list = self.motor_indices
        elif region == 'visual':
            index_list = self.visual_indices
        elif region == 'somatosensory':
            index_list = self.somatosensory_indices
        else:
            raise ValueError("Invalid region. Please specify 'motor', 'visual', or 'somatosensory'.")
            
        for idx in index_list:
            filter_1 = np.split(bandpass_filtered_data_1[idx], 17280, axis = 0)
            filter_2 = np.split(bandpass_filtered_data_2[idx], 17280, axis = 0)
            if self.complexity_calculation == 'hurst':
                hurst_1 = self.calculate_hurst_exp(filter_1)
                hurst_2 = self.calculate_hurst_exp(filter_2)
                region_ls_1.append(hurst_1)
                region_ls_2.append(hurst_2)
            elif self.complexity_calculation == 'hfd':
                hfd_1 = self.calculate_hfd_exp(filter_1)
                hfd_2 = self.calculate_hfd_exp(filter_2)
                region_ls_1.append(hfd_1)
                region_ls_2.append(hfd_2)
            else:
                raise ValueError("Invalid complexity measure")
        
        mean_array_1 = np.mean(np.array(region_ls_1), axis=0)
        mean_array_2 = np.mean(np.array(region_ls_2), axis=0)
        complexity_array = np.concatenate((mean_array_1, mean_array_2), axis=0)
        return complexity_array
    
    def one_recording(self, bandpass_filtered_data_1, region):
        
        region_ls_1 = []
        
        if region == 'motor':
            index_list = self.motor_indices
        elif region == 'visual':
            index_list = self.visual_indices
        elif region == 'somatosensory':
            index_list = self.somatosensory_indices
        else:
            raise ValueError("Invalid region. Please specify 'motor', 'visual', or 'somatosensory'.")
            
        for idx in index_list:
            filter_1 = np.split(bandpass_filtered_data_1[idx], 17280, axis = 0)
            
            if self.complexity_calculation == 'hurst':
                hurst_1 = self.calculate_hurst_exp(filter_1)
                region_ls_1.append(hurst_1)
            elif self.complexity_calculation == 'hfd':
                hfd_1 = self.calculate_hfd_exp(filter_1)
                region_ls_1.append(hfd_1)
            else:
                raise ValueError("Invalid complexity measure")
        
        mean_array = np.mean(np.array(region_ls_1), axis=0)
        return mean_array
       
        
    def calculate_hurst_exp(self, channel_data):
        hurst_values = [compute_hurst_exp(np.expand_dims(epoch, axis=0)) for epoch in channel_data]
        return hurst_values

    def calculate_hfd_exp(self, channel_data):
        hfd_values = [compute_higuchi_fd(np.expand_dims(epoch, axis=0)) for epoch in channel_data]
        return hfd_values

        