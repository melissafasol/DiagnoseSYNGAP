import os 
import sys
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit 
from imblearn.under_sampling import RandomUnderSampler

import xgboost as xgb
from xgboost.sklearn import XGBClassifier


def prepare_dataset(prepare_human_ids, SYNGAP_ls, WT_ls):
    
    hurst_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/Hurst/'
    dispen_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/DispEn/'
    phase_lock_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/Phase_Lock_Channels/'
    cross_corr_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/Cross_Corr_Channels/'
    FOOOF_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/FOOOF_all_channels/'
    delta_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/Delta_Power/'
    theta_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/Theta_Power/'
    alpha_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/Alpha_Power/'
    beta_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/Beta_Power/'
    
    patient_df_ls = []
    
    for patient in prepare_human_ids:
        
        print(patient)
        chan_O1_hurst = np.load(hurst_dir + str(patient) + '_01_hurst.npy')
        chan_C3_hurst = np.load(hurst_dir + str(patient) + '_C3_hurst.npy')
        chan_E1_hurst = np.load(hurst_dir + str(patient) + '_E1_hurst.npy')
        chan_E2_hurst = np.load(hurst_dir + str(patient) + '_E2_hurst.npy')
        chan_F3_hurst = np.load(hurst_dir + str(patient) + '_F3_hurst.npy')
        chan_M2_hurst = np.load(hurst_dir + str(patient) + '_M2_hurst.npy')
        
        chan_O1_dispen = np.load(dispen_dir + str(patient) + '_01_dispen.npy')
        chan_C3_dispen = np.load(dispen_dir + str(patient) + '_C3_dispen.npy')
        chan_E1_dispen = np.load(dispen_dir + str(patient) + '_E1_dispen.npy')
        chan_E2_dispen = np.load(dispen_dir + str(patient) + '_E2_dispen.npy')
        chan_F3_dispen = np.load(dispen_dir + str(patient) + '_F3_dispen.npy')
        chan_M2_dispen = np.load(dispen_dir + str(patient) + '_M2_dispen.npy')

        chan_E1_exp = np.load(FOOOF_dir + str(patient) + '_chan_0_exponent.npy')
        chan_E1_offset = np.load(FOOOF_dir + str(patient) + '_chan_0_offset.npy')
        chan_E2_exp = np.load(FOOOF_dir + str(patient) + '_chan_1_exponent.npy')
        chan_E2_offset = np.load(FOOOF_dir + str(patient) + '_chan_1_offset.npy')
        chan_F3_exp = np.load(FOOOF_dir + str(patient) + '_chan_2_exponent.npy')
        chan_F3_offset = np.load(FOOOF_dir + str(patient) + '_chan_2_offset.npy')
        chan_C3_exp = np.load(FOOOF_dir + str(patient) + '_chan_3_exponent.npy')
        chan_C3_offset = np.load(FOOOF_dir + str(patient) + '_chan_3_offset.npy')
        chan_O1_exp = np.load(FOOOF_dir + str(patient) + '_chan_4_exponent.npy')
        chan_O1_offset = np.load(FOOOF_dir + str(patient) + '_chan_4_offset.npy')
        chan_M2_exp = np.load(FOOOF_dir + str(patient) + '_chan_5_exponent.npy')
        chan_M2_offset = np.load(FOOOF_dir + str(patient) + '_chan_5_offset.npy')
        
        chan_E1_delta = np.load(delta_dir + str(patient) + '_chan_0.npy')
        chan_E2_delta = np.load(delta_dir + str(patient) + '_chan_1.npy')
        chan_F3_delta = np.load(delta_dir + str(patient) + '_chan_2.npy')
        chan_C3_delta = np.load(delta_dir + str(patient) + '_chan_3.npy')
        chan_O1_delta = np.load(delta_dir + str(patient) + '_chan_4.npy')
        chan_M2_delta = np.load(delta_dir + str(patient) + '_chan_5.npy')
        
        chan_E1_theta = np.load(theta_dir + str(patient) + '_chan_0.npy')
        chan_E2_theta = np.load(theta_dir + str(patient) + '_chan_1.npy')
        chan_F3_theta = np.load(theta_dir + str(patient) + '_chan_2.npy')
        chan_C3_theta = np.load(theta_dir + str(patient) + '_chan_3.npy')
        chan_O1_theta = np.load(theta_dir + str(patient) + '_chan_4.npy')
        chan_M2_theta = np.load(theta_dir + str(patient) + '_chan_5.npy')
    
        chan_E1_alpha = np.load(alpha_dir + str(patient) + '_chan_0.npy')
        chan_E2_alpha = np.load(alpha_dir + str(patient) + '_chan_1.npy')
        chan_F3_alpha = np.load(alpha_dir + str(patient) + '_chan_2.npy')
        chan_C3_alpha = np.load(alpha_dir + str(patient) + '_chan_3.npy')
        chan_O1_alpha = np.load(alpha_dir + str(patient) + '_chan_4.npy')
        chan_M2_alpha = np.load(alpha_dir + str(patient) + '_chan_5.npy')
    
        chan_E1_beta = np.load(beta_dir + str(patient) + '_chan_0.npy')
        chan_E2_beta = np.load(beta_dir + str(patient) + '_chan_1.npy')
        chan_F3_beta = np.load(beta_dir + str(patient) + '_chan_2.npy')
        chan_C3_beta = np.load(beta_dir + str(patient) + '_chan_3.npy')
        chan_O1_beta = np.load(beta_dir + str(patient) + '_chan_4.npy')
        chan_M2_beta = np.load(beta_dir + str(patient) + '_chan_5.npy')
    
        E1_E2_cross_corr = np.load(cross_corr_dir + str(patient) + '_E1_E2.npy')
        E1_F3_cross_corr = np.load(cross_corr_dir + str(patient) + '_E1_F3.npy')
        E1_C3_cross_corr = np.load(cross_corr_dir + str(patient) + '_E1_C3.npy')
        E1_O1_cross_corr = np.load(cross_corr_dir + str(patient) + '_E1_O1.npy')
        E1_M2_cross_corr = np.load(cross_corr_dir + str(patient) + '_E1_M2.npy')
        E2_F3_cross_corr = np.load(cross_corr_dir + str(patient) + '_E2_F3.npy')
        E2_C3_cross_corr = np.load(cross_corr_dir + str(patient) + '_E2_C3.npy')
        E2_O1_cross_corr = np.load(cross_corr_dir + str(patient) + '_E2_O1.npy')
        E2_M2_cross_corr = np.load(cross_corr_dir + str(patient) + '_E2_M2.npy')
        F3_C3_cross_corr = np.load(cross_corr_dir + str(patient) + '_F3_C3.npy')
        F3_O1_cross_corr = np.load(cross_corr_dir + str(patient) + '_F3_O1.npy')
        F3_M2_cross_corr = np.load(cross_corr_dir + str(patient) + '_F3_M2.npy')
        C3_O1_cross_corr = np.load(cross_corr_dir + str(patient) + '_C3_O1.npy')
        C3_M2_cross_corr = np.load(cross_corr_dir + str(patient) + '_C3_M2.npy')
        O1_M2_cross_corr = np.load(cross_corr_dir + str(patient) + '_O1_M2.npy')

        E1_E2_phase_lock = np.load(phase_lock_dir + str(patient) + '_E1_E2.npy')
        E1_F3_phase_lock = np.load(phase_lock_dir + str(patient) + '_E1_F3.npy')
        E1_C3_phase_lock = np.load(phase_lock_dir + str(patient) + '_E1_C3.npy')
        E1_O1_phase_lock = np.load(phase_lock_dir + str(patient) + '_E1_O1.npy')
        E1_M2_phase_lock = np.load(phase_lock_dir + str(patient) + '_E1_M2.npy')
        E2_F3_phase_lock = np.load(phase_lock_dir + str(patient) + '_E2_F3.npy')
        E2_C3_phase_lock = np.load(phase_lock_dir + str(patient) + '_E2_C3.npy')
        E2_O1_phase_lock = np.load(phase_lock_dir + str(patient) + '_E2_O1.npy')
        E2_M2_phase_lock = np.load(phase_lock_dir + str(patient) + '_E2_M2.npy')
        F3_C3_phase_lock = np.load(phase_lock_dir + str(patient) + '_F3_C3.npy')
        F3_O1_phase_lock = np.load(phase_lock_dir + str(patient) + '_F3_O1.npy')
        F3_M2_phase_lock = np.load(phase_lock_dir + str(patient) + '_F3_M2.npy')
        C3_O1_phase_lock = np.load(phase_lock_dir + str(patient) + '_C3_O1.npy')
        C3_M2_phase_lock = np.load(phase_lock_dir + str(patient) + '_C3_M2.npy')
        O1_M2_phase_lock = np.load(phase_lock_dir + str(patient) + '_O1_M2.npy')
        
        if patient in SYNGAP_ls:
            genotype = 1

        if patient in WT_ls:
            genotype = 0
            
   
    
        human_df = pd.DataFrame(data = {'Genotype': [genotype]*len(chan_E1_dispen), 
                  'Hurst_E1': chan_E1_hurst, 
                  'Hurst_E2': chan_E2_hurst,
                  'Hurst_O1' : chan_O1_hurst,'Hurst_C3': chan_C3_hurst,
                  'Hurst_F3': chan_F3_hurst, 
                  'Hurst_M2': chan_M2_hurst,
                  
                  'Dispen_E1': chan_E1_dispen,
                  #'Dispen_E2': chan_E2_dispen,
                  #'Dispen_O1' : chan_O1_dispen,
                  # 'Dispen_C3': chan_C3_dispen,
                  #'Dispen_F3': chan_F3_dispen, 
                  'Dispen_M2': chan_M2_dispen,
                  
                  #'Exp_E1': chan_E1_exp, 
                  #'Exp_E2': chan_E2_exp,
                  #'Exp_O1' : chan_O1_exp,
                  'Exp_C3': chan_C3_exp,
                  'Exp_F3': chan_F3_exp, 
                  'Exp_M2': chan_M2_exp,
                  
                  'Off_E1': chan_E1_offset, 
                  'Off_E2': chan_E2_offset,
                  #'Off_O1' : chan_O1_offset,                  
                  #'Off_C3': chan_C3_offset,
                  'Off_F3': chan_F3_offset, 
                  #'Off_M2': chan_M2_offset,
                  
                  #'delta_E1': chan_E1_delta,
                  #'delta_E2': chan_E2_delta,
                  'delta_O1' : chan_O1_delta,
                  'delta_C3': chan_C3_delta,
                  #'delta_F3': chan_F3_delta, 
                  # 'delta_M2': chan_M2_delta,
                  
                  'theta_E1': chan_E1_theta, 'delta_E2': chan_E2_theta,
                  'theta_O1' : chan_O1_theta,'delta_C3': chan_C3_theta,
                  'theta_F3': chan_F3_theta, 'delta_M2': chan_M2_theta,
                  
                  'alpha_E1': chan_E1_alpha, 'alpha_E2': chan_E2_alpha,
                  'alpha_O1' : chan_O1_alpha,'alpha_C3': chan_C3_alpha,
                  'alpha_F3': chan_F3_alpha, 'alpha_M2': chan_M2_alpha,
                  
                  'beta_E1': chan_E1_beta, 'delta_E2': chan_E2_beta,
                  'beta_O1' : chan_O1_beta,'delta_C3': chan_C3_beta,
                  'beta_F3': chan_F3_beta, 'delta_M2': chan_M2_beta,
                    
                  'E1_E2_cross_corr': E1_E2_cross_corr, 'E1_F3_cross_corr': E1_F3_cross_corr,
                  'E1_C3_cross_corr': E1_C3_cross_corr, 
                  'E1_O1_cross_corr': E1_O1_cross_corr,
                  'E1_M2_cross_corr': E1_M2_cross_corr, 'E2_F3_cross_corr': E2_F3_cross_corr,
                  'E2_C3_cross_corr': E2_C3_cross_corr, 
                  'E2_O1_cross_corr': E2_O1_cross_corr,
                  'E2_M2_cross_corr': E2_M2_cross_corr, 'F3_C3_cross_corr': F3_C3_cross_corr,
                  'F3_O1_cross_corr': F3_O1_cross_corr, 'F3_M2_cross_corr': F3_M2_cross_corr,
                  'C3_O1_cross_corr': C3_O1_cross_corr, 'C3_M2_cross_corr': C3_M2_cross_corr,
                  'O1_M2_cross_corr': O1_M2_cross_corr,
                 
                  'E1_E2_phase_lock': E1_E2_phase_lock, 'E1_F3_phase_lock': E1_F3_phase_lock,
                  'E1_C3_phase_lock': E1_C3_phase_lock, 'E1_O1_phase_lock': E1_O1_phase_lock,
                  'E1_M2_phase_lock': E1_M2_phase_lock, 'E2_F3_phase_lock': E2_F3_phase_lock,
                  'E2_C3_phase_lock': E2_C3_phase_lock, 
                                    
                                    
                  'E2_O1_phase_lock': E2_O1_phase_lock,
                  'E2_M2_phase_lock': E2_M2_phase_lock, 'F3_C3_phase_lock': F3_C3_phase_lock,
                  'F3_O1_phase_lock': F3_O1_phase_lock, 
                  'F3_M2_phase_lock': F3_M2_phase_lock,
                  'C3_O1_phase_lock': C3_O1_phase_lock, 'C3_M2_phase_lock': C3_M2_phase_lock,
                  'O1_M2_phase_lock': O1_M2_phase_lock})
  
        #print(human_dict)
        #human_df = pd.DataFrame(data = human_dict) 
        patient_df_ls.append(human_df)
        
    df_concat = pd.concat(patient_df_ls)
    
    return df_concat

def prepare_dataset_nofs(prepare_human_ids, SYNGAP_ls, WT_ls):
    
    hurst_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/Hurst/'
    dispen_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/DispEn/'
    phase_lock_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/Phase_Lock_Channels/'
    cross_corr_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/Cross_Corr_Channels/'
    FOOOF_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/FOOOF_all_channels/'
    delta_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/Delta_Power/'
    theta_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/Theta_Power/'
    alpha_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/Alpha_Power/'
    beta_dir = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/Beta_Power/'
    
    patient_df_ls = []
    
    for patient in prepare_human_ids:
        
        print(patient)
        chan_O1_hurst = np.load(hurst_dir + str(patient) + '_01_hurst.npy')
        chan_C3_hurst = np.load(hurst_dir + str(patient) + '_C3_hurst.npy')
        chan_E1_hurst = np.load(hurst_dir + str(patient) + '_E1_hurst.npy')
        chan_E2_hurst = np.load(hurst_dir + str(patient) + '_E2_hurst.npy')
        chan_F3_hurst = np.load(hurst_dir + str(patient) + '_F3_hurst.npy')
        chan_M2_hurst = np.load(hurst_dir + str(patient) + '_M2_hurst.npy')
        
        chan_O1_dispen = np.load(dispen_dir + str(patient) + '_01_dispen.npy')
        chan_C3_dispen = np.load(dispen_dir + str(patient) + '_C3_dispen.npy')
        chan_E1_dispen = np.load(dispen_dir + str(patient) + '_E1_dispen.npy')
        chan_E2_dispen = np.load(dispen_dir + str(patient) + '_E2_dispen.npy')
        chan_F3_dispen = np.load(dispen_dir + str(patient) + '_F3_dispen.npy')
        chan_M2_dispen = np.load(dispen_dir + str(patient) + '_M2_dispen.npy')

        chan_E1_exp = np.load(FOOOF_dir + str(patient) + '_chan_0_exponent.npy')
        chan_E1_offset = np.load(FOOOF_dir + str(patient) + '_chan_0_offset.npy')
        chan_E2_exp = np.load(FOOOF_dir + str(patient) + '_chan_1_exponent.npy')
        chan_E2_offset = np.load(FOOOF_dir + str(patient) + '_chan_1_offset.npy')
        chan_F3_exp = np.load(FOOOF_dir + str(patient) + '_chan_2_exponent.npy')
        chan_F3_offset = np.load(FOOOF_dir + str(patient) + '_chan_2_offset.npy')
        chan_C3_exp = np.load(FOOOF_dir + str(patient) + '_chan_3_exponent.npy')
        chan_C3_offset = np.load(FOOOF_dir + str(patient) + '_chan_3_offset.npy')
        chan_O1_exp = np.load(FOOOF_dir + str(patient) + '_chan_4_exponent.npy')
        chan_O1_offset = np.load(FOOOF_dir + str(patient) + '_chan_4_offset.npy')
        chan_M2_exp = np.load(FOOOF_dir + str(patient) + '_chan_5_exponent.npy')
        chan_M2_offset = np.load(FOOOF_dir + str(patient) + '_chan_5_offset.npy')
        
        chan_E1_delta = np.load(delta_dir + str(patient) + '_chan_0.npy')
        chan_E2_delta = np.load(delta_dir + str(patient) + '_chan_1.npy')
        chan_F3_delta = np.load(delta_dir + str(patient) + '_chan_2.npy')
        chan_C3_delta = np.load(delta_dir + str(patient) + '_chan_3.npy')
        chan_O1_delta = np.load(delta_dir + str(patient) + '_chan_4.npy')
        chan_M2_delta = np.load(delta_dir + str(patient) + '_chan_5.npy')
        
        chan_E1_theta = np.load(theta_dir + str(patient) + '_chan_0.npy')
        chan_E2_theta = np.load(theta_dir + str(patient) + '_chan_1.npy')
        chan_F3_theta = np.load(theta_dir + str(patient) + '_chan_2.npy')
        chan_C3_theta = np.load(theta_dir + str(patient) + '_chan_3.npy')
        chan_O1_theta = np.load(theta_dir + str(patient) + '_chan_4.npy')
        chan_M2_theta = np.load(theta_dir + str(patient) + '_chan_5.npy')
    
        chan_E1_alpha = np.load(alpha_dir + str(patient) + '_chan_0.npy')
        chan_E2_alpha = np.load(alpha_dir + str(patient) + '_chan_1.npy')
        chan_F3_alpha = np.load(alpha_dir + str(patient) + '_chan_2.npy')
        chan_C3_alpha = np.load(alpha_dir + str(patient) + '_chan_3.npy')
        chan_O1_alpha = np.load(alpha_dir + str(patient) + '_chan_4.npy')
        chan_M2_alpha = np.load(alpha_dir + str(patient) + '_chan_5.npy')
    
        chan_E1_beta = np.load(beta_dir + str(patient) + '_chan_0.npy')
        chan_E2_beta = np.load(beta_dir + str(patient) + '_chan_1.npy')
        chan_F3_beta = np.load(beta_dir + str(patient) + '_chan_2.npy')
        chan_C3_beta = np.load(beta_dir + str(patient) + '_chan_3.npy')
        chan_O1_beta = np.load(beta_dir + str(patient) + '_chan_4.npy')
        chan_M2_beta = np.load(beta_dir + str(patient) + '_chan_5.npy')
    
        E1_E2_cross_corr = np.load(cross_corr_dir + str(patient) + '_E1_E2.npy')
        E1_F3_cross_corr = np.load(cross_corr_dir + str(patient) + '_E1_F3.npy')
        E1_C3_cross_corr = np.load(cross_corr_dir + str(patient) + '_E1_C3.npy')
        E1_O1_cross_corr = np.load(cross_corr_dir + str(patient) + '_E1_O1.npy')
        E1_M2_cross_corr = np.load(cross_corr_dir + str(patient) + '_E1_M2.npy')
        E2_F3_cross_corr = np.load(cross_corr_dir + str(patient) + '_E2_F3.npy')
        E2_C3_cross_corr = np.load(cross_corr_dir + str(patient) + '_E2_C3.npy')
        E2_O1_cross_corr = np.load(cross_corr_dir + str(patient) + '_E2_O1.npy')
        E2_M2_cross_corr = np.load(cross_corr_dir + str(patient) + '_E2_M2.npy')
        F3_C3_cross_corr = np.load(cross_corr_dir + str(patient) + '_F3_C3.npy')
        F3_O1_cross_corr = np.load(cross_corr_dir + str(patient) + '_F3_O1.npy')
        F3_M2_cross_corr = np.load(cross_corr_dir + str(patient) + '_F3_M2.npy')
        C3_O1_cross_corr = np.load(cross_corr_dir + str(patient) + '_C3_O1.npy')
        C3_M2_cross_corr = np.load(cross_corr_dir + str(patient) + '_C3_M2.npy')
        O1_M2_cross_corr = np.load(cross_corr_dir + str(patient) + '_O1_M2.npy')

        E1_E2_phase_lock = np.load(phase_lock_dir + str(patient) + '_E1_E2.npy')
        E1_F3_phase_lock = np.load(phase_lock_dir + str(patient) + '_E1_F3.npy')
        E1_C3_phase_lock = np.load(phase_lock_dir + str(patient) + '_E1_C3.npy')
        E1_O1_phase_lock = np.load(phase_lock_dir + str(patient) + '_E1_O1.npy')
        E1_M2_phase_lock = np.load(phase_lock_dir + str(patient) + '_E1_M2.npy')
        E2_F3_phase_lock = np.load(phase_lock_dir + str(patient) + '_E2_F3.npy')
        E2_C3_phase_lock = np.load(phase_lock_dir + str(patient) + '_E2_C3.npy')
        E2_O1_phase_lock = np.load(phase_lock_dir + str(patient) + '_E2_O1.npy')
        E2_M2_phase_lock = np.load(phase_lock_dir + str(patient) + '_E2_M2.npy')
        F3_C3_phase_lock = np.load(phase_lock_dir + str(patient) + '_F3_C3.npy')
        F3_O1_phase_lock = np.load(phase_lock_dir + str(patient) + '_F3_O1.npy')
        F3_M2_phase_lock = np.load(phase_lock_dir + str(patient) + '_F3_M2.npy')
        C3_O1_phase_lock = np.load(phase_lock_dir + str(patient) + '_C3_O1.npy')
        C3_M2_phase_lock = np.load(phase_lock_dir + str(patient) + '_C3_M2.npy')
        O1_M2_phase_lock = np.load(phase_lock_dir + str(patient) + '_O1_M2.npy')
        
        if patient in SYNGAP_ls:
            genotype = 1

        if patient in WT_ls:
            genotype = 0
            
   
    
        human_df = pd.DataFrame(data = {'Genotype': [genotype]*len(chan_E1_dispen), 
                  'Hurst_E1': chan_E1_hurst, 
                  'Hurst_E2': chan_E2_hurst,
                  'Hurst_O1' : chan_O1_hurst,'Hurst_C3': chan_C3_hurst,
                  'Hurst_F3': chan_F3_hurst, 
                  'Hurst_M2': chan_M2_hurst,
                  
                  'Dispen_E1': chan_E1_dispen,
                  'Dispen_E2': chan_E2_dispen,
                  'Dispen_O1' : chan_O1_dispen,'Dispen_C3': chan_C3_dispen,
                  'Dispen_F3': chan_F3_dispen, 
                  'Dispen_M2': chan_M2_dispen,
                  
                  'Exp_E1': chan_E1_exp, 
                  'Exp_E2': chan_E2_exp,
                  'Exp_O1' : chan_O1_exp,'Exp_C3': chan_C3_exp,
                  'Exp_F3': chan_F3_exp, 'Exp_M2': chan_M2_exp,
                  
                  'Off_E1': chan_E1_offset, 'Off_E2': chan_E2_offset,
                  'Off_O1' : chan_O1_offset,                  
                  'Off_C3': chan_C3_offset,
                  'Off_F3': chan_F3_offset, 
                  'Off_M2': chan_M2_offset,
                  
                  'delta_E1': chan_E1_delta,
                  'delta_E2': chan_E2_delta,
                  'delta_O1' : chan_O1_delta,
                  'delta_C3': chan_C3_delta,
                  'delta_F3': chan_F3_delta, 'delta_M2': chan_M2_delta,
                  
                  'theta_E1': chan_E1_theta, 'delta_E2': chan_E2_theta,
                  'theta_O1' : chan_O1_theta,'delta_C3': chan_C3_theta,
                  'theta_F3': chan_F3_theta, 'delta_M2': chan_M2_theta,
                  
                  'alpha_E1': chan_E1_alpha, 'alpha_E2': chan_E2_alpha,
                  'alpha_O1' : chan_O1_alpha,'alpha_C3': chan_C3_alpha,
                  'alpha_F3': chan_F3_alpha, 'alpha_M2': chan_M2_alpha,
                  
                  'beta_E1': chan_E1_beta, 'delta_E2': chan_E2_beta,
                  'beta_O1' : chan_O1_beta,'delta_C3': chan_C3_beta,
                  'beta_F3': chan_F3_beta, 'delta_M2': chan_M2_beta,
                    
                  'E1_E2_cross_corr': E1_E2_cross_corr, 'E1_F3_cross_corr': E1_F3_cross_corr,
                  'E1_C3_cross_corr': E1_C3_cross_corr, 
                  'E1_O1_cross_corr': E1_O1_cross_corr,
                  'E1_M2_cross_corr': E1_M2_cross_corr, 'E2_F3_cross_corr': E2_F3_cross_corr,
                  'E2_C3_cross_corr': E2_C3_cross_corr, 
                  'E2_O1_cross_corr': E2_O1_cross_corr,
                  'E2_M2_cross_corr': E2_M2_cross_corr, 'F3_C3_cross_corr': F3_C3_cross_corr,
                  'F3_O1_cross_corr': F3_O1_cross_corr, 'F3_M2_cross_corr': F3_M2_cross_corr,
                  'C3_O1_cross_corr': C3_O1_cross_corr, 'C3_M2_cross_corr': C3_M2_cross_corr,
                  'O1_M2_cross_corr': O1_M2_cross_corr,
                 
                  'E1_E2_phase_lock': E1_E2_phase_lock, 'E1_F3_phase_lock': E1_F3_phase_lock,
                  'E1_C3_phase_lock': E1_C3_phase_lock, 'E1_O1_phase_lock': E1_O1_phase_lock,
                  'E1_M2_phase_lock': E1_M2_phase_lock, 'E2_F3_phase_lock': E2_F3_phase_lock,
                  'E2_C3_phase_lock': E2_C3_phase_lock, 
                                    
                                    
                  'E2_O1_phase_lock': E2_O1_phase_lock,
                  'E2_M2_phase_lock': E2_M2_phase_lock, 'F3_C3_phase_lock': F3_C3_phase_lock,
                  'F3_O1_phase_lock': F3_O1_phase_lock, 
                  'F3_M2_phase_lock': F3_M2_phase_lock,
                  'C3_O1_phase_lock': C3_O1_phase_lock, 'C3_M2_phase_lock': C3_M2_phase_lock,
                  'O1_M2_phase_lock': O1_M2_phase_lock})
  
        #print(human_dict)
        #human_df = pd.DataFrame(data = human_dict) 
        patient_df_ls.append(human_df)
        
    df_concat = pd.concat(patient_df_ls)
    
    return df_concat