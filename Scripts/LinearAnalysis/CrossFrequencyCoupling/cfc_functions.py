'''Script to apply method from Multivariate cross-frequency coupling via generalized eigendecomposition paper'''

import os 
import sys 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import signal
from neurodsp.utils import create_times


from preprocessing import FilterCFC, ExtractThetaTroughIndices


class CrossFreqCouplingCalculations():
    
    def __init__(self, seizure_array, start_time_seiz, end_time_seiz, length_size):
        '''seizure array (16, 1252): one seizure epoch, 
        start_time_seiz: seizure start time 
        end_time_seiz: seizure end time
        lenght theta trough: lenght of seizure epoch
        channel: channel to use for theta trough extraction'''
        self.seizure_array = seizure_array
        self.start_time_seiz = start_time_seiz
        self.end_time_seiz = end_time_seiz 
        self.length_size = length_size

        
    def filter_gamma_theta(self,):
        gf = FilterCFC(self.seizure_array, filter_type = 'gamma')
        gamma_filtered_data = gf.butter_bandpass()
        tf = FilterCFC(self.seizure_array, filter_type = 'theta')
        theta_filtered_data = tf.butter_bandpass()
        
        return gamma_filtered_data, theta_filtered_data
        
    def extract_theta_trough(self,theta_filtered_data):
        extract_theta_trough = ExtractThetaTroughIndices(theta_filtered_data, self.start_time_seiz, self.end_time_seiz,
                                                         length_theta_trough = (0, 5))
        channel_theta_trough = extract_theta_trough.apply_moving_mean()
        theta_trough, pre_trough, post_trough = extract_theta_trough.extract_trough_indices(channel_theta_trough)
        return theta_trough, pre_trough, post_trough
    
    def theta_trough_raw(self, pre_trough, post_trough, unfiltered_seizure_array):
        '''input list of pre_trough and post-trough to extract theta trough and extract the corresponding
        time indices from the seizure array'''
        theta_raw_list = []
        for idx_pre, idx_post in zip(pre_trough, post_trough):
            test = unfiltered_seizure_array[:, idx_pre[0]: idx_post[1]]
            theta_raw_list.append(test)
            
        theta_raw_signals = np.concatenate(theta_raw_list, axis = 1)
        return theta_raw_signals
    
    def cov_matrices(self, gamma_ref_data, theta_signal_data):
        '''input two arrays - one array with the signal data and one with the reference data'''
        
        cov_gamma = np.cov(gamma_ref_data)
        cov_theta = np.cov(theta_signal_data)
        
        return cov_gamma, cov_theta 
        
    def eigenvalue_decomposition(self, cov_gamma, cov_theta):
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
    
    def component_time_series(self, maximum_v, gamma_filtered_data):
        '''multiply each of the time series by corresponding element in eigenvector and add of these up 
        (chan_1 * first element of v vector + chan_2 * second element of v vector.....the end result 
        is one single time series that is like a component time series'''
        transpose_time = gamma_filtered_data.transpose()
        weighted_time = transpose_time*maximum_v
        
        #sum rows 
        weighted_time_series = np.sum(weighted_time, axis = 1)
        return weighted_time_series
    