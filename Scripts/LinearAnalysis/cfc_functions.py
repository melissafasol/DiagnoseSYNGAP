'''Script to apply method from Multivariate cross-frequency coupling via generalized eigendecomposition paper'''

import os 
import sys 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import signal
from neurodsp.utils import create_times

from preprocessing import FilterCFC, ExtractThetaTroughIndices


class FilterCFC:
    
    order = 3
    sampling_rate = 250.4
    nyquist = 125.2
    
    def __init__(self, unfiltered_data, filter_type):
        self.unfiltered_data = unfiltered_data
        self.filter_type = filter_type
        if self.filter_type == 'norm':
            self.low = 0.2/self.nyquist
            self.high = 48/self.nyquist
        elif self.filter_type == 'theta':
            self.low = 5/self.nyquist
            self.high = 10/self.nyquist
        elif self.filter_type == 'gamma':
            self.low = 30/self.nyquist
            self.high = 48/self.nyquist
    
    def butter_bandpass(self):
        butter_b, butter_a = signal.butter(self.order, [self.low, self.high], btype='band', analog = False)
        
        filtered_data = signal.filtfilt(butter_b, butter_a, self.unfiltered_data)
        
        return filtered_data


def theta_trough_raw(pre_trough, post_trough, unfiltered_seizure_array):
    '''input list of pre_trough and post-trough to extract theta trough and extract the corresponding
    time indices from the seizure array'''
    theta_raw_list = []
    for idx_pre, idx_post in zip(pre_trough, post_trough):
        test = unfiltered_seizure_array[:, idx_pre[0]: idx_post[1]]
        theta_raw_list.append(test)
            
    theta_raw_signals = np.concatenate(theta_raw_list, axis = 1)
    return theta_raw_signals
    
def cov_matrices(gamma_ref_data, theta_signal_data):
    '''input two arrays - one array with the signal data and one with the reference data'''
        
    cov_gamma = np.cov(gamma_ref_data)
    cov_theta = np.cov(theta_signal_data)
        
    return cov_gamma, cov_theta 
        
def eigenvalue_decomposition(cov_gamma, cov_theta):
    '''returns first column of eigenvector - this is a set of channel weights that maximally differentiates the signal and reference'''
    #calc the inverse of the reference matrix 
    Rinv = np.linalg.inv(cov_gamma)
    S = cov_theta
    #calc the matrix product of Rinv and S
    matrix_product = np.matmul(Rinv, S)
    #generalised eigendecomposition of matrix product 
    lambdas, v = np.linalg.eig(matrix_product)
    #the first column of v has the largest associated eigenvalue and is the one that maximally differentiates the two matrices
    v_tranpose = v.transpose()
    maximum_v = v_tranpose[0]
        
    return maximum_v 
    
def component_time_series(maximum_v, gamma_filtered_data):
    '''multiply each of the time series by corresponding element in eigenvector and add of these up 
    (chan_1 * first element of v vector + chan_2 * second element of v vector.....the end result 
    is one single time series that is like a component time series'''
    transpose_time = gamma_filtered_data.transpose()
    weighted_time = transpose_time*maximum_v
        
    #sum rows 
    weighted_time_series = np.sum(weighted_time, axis = 1)
    return weighted_time_series

def extract_theta_trough(raw_data, troughs):
    
    theta_trough_ls = []
    
    for trough_idx in troughs:
        pre_trough = trough_idx - 6
        trough_end = trough_idx + 6
        theta_trough_ls.append(raw_data[:, pre_trough: trough_end + 1])
    
    check_len = []
    for epoch in theta_trough_ls:
        if len(epoch[0]) == len(theta_trough_ls[0][0]):
            check_len.append(epoch)
        else:
            pass
    
    theta_trough_raw = np.concatenate(check_len, axis = 1)
    return theta_trough_raw
        