import os 
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, KFold
from xgboost import XGBClassifier

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GroupShuffleSplit

import xgboost as xgb
import xgbfir
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import dtreeviz
from typing import Any, Dict, Union

from yellowbrick import model_selection as ms
from yellowbrick.model_selection import validation_curve

from sklearn import metrics


from dataset_prep import prepare_df_1, prepare_df_2
from hyperparam_hyperopt import hyperparameter_tuning

br_directory = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline/'
motor_hurst_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Hurst/'
motor_hfd_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1//Motor/HFD/'
motor_dispen_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/DispEn/'
motor_gamma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Gamma_Power/'
motor_theta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Theta_Power/'

soma_hurst_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Hurst/'
soma_hfd_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/HFD/'
soma_dispen_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/DispEn/'
soma_gamma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Gamma_Power/'
soma_theta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Theta_Power/'

vis_hurst_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Hurst/'
vis_hfd_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/HFD/'
vis_dispen_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/DispEn/'
vis_gamma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Gamma_Power/'
vis_theta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Theta_Power/'

#connectivity indices
cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/CrossCorr/'
mot_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/CrossCorr_Motor/' 
som_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/CrossCorr_Somatosensory/'
vis_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/CrossCorr_Visual/'

mot_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Phase_Lock_Motor/' 
som_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Phase_Lock_Somato/'
vis_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Phase_Lock_Visual/'

train_2_ids = ['S7070', 'S7072', 'S7083', 'S7064', 'S7096']#, 'S7069', 'S7091']
train_1_ids = ['S7088', 'S7092', 'S7094', 'S7075', 'S7071'] #, 'S7076', 'S7101']
val_2_ids = ['S7069', 'S7091']
val_1_ids = ['S7076', 'S7101']
test_2_ids = ['S7086'] # 'S7063'
test_1_ids = ['S7076'] #'S7074'

SYNGAP_het = ['S7063', 'S7064', 'S7069', 'S7072', 'S7075', 'S7076', 'S7088', 'S7092', 'S7094', 'S7096']
SYNGAP_wt = ['S7068', 'S7070', 'S7071', 'S7074', 'S7083', 'S7086', 'S7091', 'S7098', 'S7101']

seizure_two_syngap = ['S7063', 'S7064', 'S7069', 'S7072']
seizure_one_syngap = ['S7075', 'S7092', 'S7094']


feature_df_2_ids = []
for animal in train_2_ids:
    print(animal)
    #load br file 
    br_1 = pd.read_pickle(br_directory + str(animal) + '_BL1.pkl')
    br_2 = pd.read_pickle(br_directory + str(animal) + '_BL2.pkl')
    br_state_1 = br_1['brainstate'].to_numpy()
    br_state_2 = br_2['brainstate'].to_numpy()
    br_state = np.concatenate([br_state_1, br_state_2])
    
    #motor 
    motor_hfd = np.load(motor_hfd_dir + animal + '_hfd_concat.npy')
    motor_hfd_avg = [value[0] for value in motor_hfd]
    motor_hurst = np.load(motor_hurst_dir + animal + '_hurst_concat.npy')
    motor_hurst_avg = [value[0] for value in motor_hurst]
    
    motor_dispen = np.load(motor_dispen_dir + animal + '_dispen.npy')
    motor_gamma = np.load(motor_gamma_dir + animal + '_power.npy') 
    motor_theta = np.load(motor_theta_dir + animal + '_power.npy') 
    
    #somatosensory 
    soma_hfd = np.load(soma_hfd_dir + animal + '_hfd_concat.npy')
    soma_hfd_avg = [value[0] for value in soma_hfd]
    soma_hurst = np.load(soma_hurst_dir + animal + '_hurst_concat.npy')
    soma_hurst_avg = [value[0] for value in soma_hurst]
    
    soma_dispen = np.load(soma_dispen_dir + animal + '_dispen.npy')
    soma_gamma = np.load(soma_gamma_dir + animal + '_power.npy') 
    soma_theta = np.load(soma_theta_dir + animal + '_power.npy') 
    
    
    #somatosensory 
    vis_hfd = np.load(vis_hfd_dir + animal + '_hfd_concat.npy')
    vis_hfd_avg = [value[0] for value in vis_hfd]
    vis_hurst = np.load(vis_hurst_dir + animal + '_hurst_concat.npy')
    vis_hurst_avg = [value[0] for value in vis_hurst]
    
    vis_dispen = np.load(vis_dispen_dir + animal + '_dispen.npy')
    vis_gamma = np.load(vis_gamma_dir + animal + '_power.npy') 
    vis_theta = np.load(vis_theta_dir + animal + '_power.npy')
    
    #cross cor
    mot_cross_corr_left = np.load(mot_cross_cor_dir + str(animal) + '_mot_left_cross_cor.npy')
    mot_cross_corr_right = np.load(mot_cross_cor_dir + str(animal) + '_mot_right_cross_cor.npy')
    som_cross_corr_left = np.load(som_cross_cor_dir + str(animal) + '_som_left_cross_cor.npy')
    som_cross_corr_right = np.load(som_cross_cor_dir + str(animal) + '_som_right_cross_cor.npy')
    vis_cross_corr_left = np.load(vis_cross_cor_dir + str(animal) + '_vis_left_cross_cor.npy')
    vis_cross_corr_right = np.load(vis_cross_cor_dir + str(animal) + '_vis_right_cross_cor.npy')
    
    #phase lock 
    mot_phase_lock_left = np.load(mot_phase_lock_dir + str(animal) + '_mot_left_phase_lock.npy')
    mot_phase_lock_right = np.load(mot_phase_lock_dir + str(animal) + '_mot_right_phase_lock.npy')
    som_phase_lock_left = np.load(som_phase_lock_dir + str(animal) + '_som_left_phase_lock.npy')
    som_phase_lock_right = np.load(som_phase_lock_dir + str(animal) + '_som_right_phase_lock.npy')
    vis_phase_lock_left = np.load(vis_phase_lock_dir + str(animal) + '_vis_left_phase_lock.npy')
    vis_phase_lock_right = np.load(vis_phase_lock_dir + str(animal) + '_vis_right_phase_lock.npy')
    
    #cross_corr_errors
    error_1 = np.load(cross_cor_dir + animal + '_error_br_1.npy')
    error_2 = np.load(cross_cor_dir + animal + '_error_br_2.npy')
    
    
    if len(error_1) > 0 and len(error_2) >0:
        print(animal + ' error')
        error_2_correct = error_2 + 17280
        errors = np.concatenate([error_1, error_2_correct])
        br_state = np.delete(br_state, errors)
        motor_hfd_avg = np.delete(motor_hfd_avg, errors)
        motor_hurst_avg = np.delete(motor_hurst_avg, errors)
        motor_dispen = np.delete(motor_dispen, errors)
        motor_gamma = np.delete(motor_gamma, errors)
        motor_theta = np.delete(motor_theta, errors)
        soma_hfd_avg = np.delete(soma_hfd_avg, errors)
        soma_hurst_avg = np.delete(soma_hurst_avg, errors)
        soma_dispen = np.delete(soma_dispen, errors)
        soma_gamma = np.delete(soma_gamma, errors)
        soma_theta = np.delete(soma_theta, errors)
        vis_hfd_avg = np.delete(vis_hfd_avg, errors)
        vis_hurst_avg = np.delete(vis_hurst_avg, errors)
        vis_dispen = np.delete(vis_dispen, errors)
        vis_gamma = np.delete(vis_gamma, errors)
        vis_theta = np.delete(vis_theta, errors)
        mot_phase_lock_left = np.delete(mot_phase_lock_left, errors)
        mot_phase_lock_right = np.delete(mot_phase_lock_right, errors)
        som_phase_lock_left = np.delete(som_phase_lock_left, errors)
        som_phase_lock_right = np.delete(som_phase_lock_right, errors)
        vis_phase_lock_left = np.delete(vis_phase_lock_left, errors)
        vis_phase_lock_right = np.delete(vis_phase_lock_right, errors)
        
    elif len(error_1) > 0:
        br_state = np.delete(br_state, error_1)
        motor_hfd_avg = np.delete(motor_hfd_avg, error_1)
        motor_hurst_avg = np.delete(motor_hurst_avg, error_1)
        motor_dispen = np.delete(motor_dispen, error_1)
        motor_gamma = np.delete(motor_gamma, error_1)
        motor_theta = np.delete(motor_theta, error_1)
        soma_hfd_avg = np.delete(soma_hfd_avg, error_1)
        soma_hurst_avg = np.delete(soma_hurst_avg, error_1)
        soma_dispen = np.delete(soma_dispen, error_1)
        soma_gamma = np.delete(soma_gamma, error_1)
        soma_theta = np.delete(soma_theta, error_1)
        vis_hfd_avg = np.delete(vis_hfd_avg, error_1)
        vis_hurst_avg = np.delete(vis_hurst_avg, error_1)
        vis_dispen = np.delete(vis_dispen, error_1)
        vis_gamma = np.delete(vis_gamma, error_1)
        vis_theta = np.delete(vis_theta, error_1)
        mot_phase_lock_left = np.delete(mot_phase_lock_left, error_1)
        mot_phase_lock_right = np.delete(mot_phase_lock_right, error_1)
        som_phase_lock_left = np.delete(som_phase_lock_left, error_1)
        som_phase_lock_right = np.delete(som_phase_lock_right, error_1)
        vis_phase_lock_left = np.delete(vis_phase_lock_left, error_1)
        vis_phase_lock_right = np.delete(vis_phase_lock_right, error_1)
    elif len(error_2) > 0:
        print(animal + ' error 2')
        error_2_br_2 = error_2 + 17280
        br_state = np.delete(br_state, error_2_br_2)
        motor_hfd_avg = np.delete(motor_hfd_avg, error_2_br_2)
        motor_hurst_avg = np.delete(motor_hurst_avg, error_2_br_2)
        motor_dispen = np.delete(motor_dispen, error_2_br_2)
        motor_gamma = np.delete(motor_gamma, error_2_br_2)
        motor_theta = np.delete(motor_theta, error_2_br_2)
        soma_hfd_avg = np.delete(soma_hfd_avg, error_2_br_2)
        soma_hurst_avg = np.delete(soma_hurst_avg, error_2_br_2)
        soma_dispen = np.delete(soma_dispen, error_2_br_2)
        soma_gamma = np.delete(soma_gamma, error_2_br_2)
        soma_theta = np.delete(soma_theta, error_2_br_2)
        vis_hfd_avg = np.delete(vis_hfd_avg, error_2_br_2)
        vis_hurst_avg = np.delete(vis_hurst_avg, error_2_br_2)
        vis_dispen = np.delete(vis_dispen, error_2_br_2)
        vis_gamma = np.delete(vis_gamma, error_2_br_2)
        vis_theta = np.delete(vis_theta, error_2_br_2)
        mot_phase_lock_left = np.delete(mot_phase_lock_left, error_2_br_2)
        mot_phase_lock_right = np.delete(mot_phase_lock_right, error_2_br_2)
        som_phase_lock_left = np.delete(som_phase_lock_left, error_2_br_2)
        som_phase_lock_right = np.delete(som_phase_lock_right, error_2_br_2)
        vis_phase_lock_left = np.delete(vis_phase_lock_left, error_2_br_2)
        vis_phase_lock_right = np.delete(vis_phase_lock_right, error_2_br_2)
    else:
        pass
    
    
    
     #clean arrays
    #clean_offset = np.delete(fooof_offset_nan, nan_indices)
    #clean_exponent = np.delete(fooof_exponent_nan, nan_indices)
    #clean_dispen = np.delete(dispen, nan_indices)
    #clean_gamma = np.delete(gamma, nan_indices)
    #clean_br_state = np.delete(br_state, nan_indices)
    
    
    if animal in SYNGAP_het:
        genotype = 1
    elif animal in SYNGAP_wt:
        genotype = 0
        

    region_dict = {'Genotype': [genotype]*len(motor_dispen), 'SleepStage': br_state,
                   'Motor_DispEn': motor_dispen, 'Motor_Hurst': motor_hurst_avg, 
                   'Motor_HFD': motor_hfd_avg,'Motor_Gamma': motor_gamma,
                   'Motor_Theta': motor_theta,
                   'Soma_DispEn': soma_dispen,'Soma_Hurst': soma_hurst_avg,
                   'Soma_HFD': soma_hfd_avg,'Soma_Gamma': soma_gamma,
                   'Soma_Theta': soma_theta,
                   'Visual_DispEn': vis_dispen,'Visual_Hurst': vis_hurst_avg,
                   'Visual_HFD': vis_hfd_avg, 'Vis_Gamma': vis_gamma, 
                   'Vis_Theta': vis_theta,
                   'Mot_CC_Right': mot_cross_corr_right, 'Mot_CC_Left': mot_cross_corr_left,
                   'Som_CC_Right': som_cross_corr_right, 'Som_CC_Left': som_cross_corr_left,
                   'Vis_CC_Right': vis_cross_corr_right, 'Mot_CC_Left': vis_cross_corr_left,
                   'Mot_PL_Right': mot_phase_lock_right, 'Mot_PL_Left': mot_phase_lock_left,
                   'Som_PL_Right': som_phase_lock_right, 'Som_PL_Left': som_phase_lock_left,
                   'Vis_PL_Right': vis_phase_lock_right, 'Vis_PL_Left': vis_phase_lock_left}
    
    
    region_df = pd.DataFrame(data = region_dict)
    clean_df = region_df[region_df["SleepStage"].isin([0,1,2])]
    print(len(clean_df))
    print(clean_df)
    feature_df_2_ids.append(clean_df)
    
feature_df_1_ids = []
for animal in train_1_ids:
    print(animal)
    #load br file 
    br_1 = pd.read_pickle(br_directory + str(animal) + '_BL1.pkl')
    br_state = br_1['brainstate'].to_numpy()
    
    #motor 
    motor_hfd = np.load(motor_hfd_dir + animal + '_hfd_concat.npy')
    motor_hfd_avg = [value[0] for value in motor_hfd]
    motor_hurst = np.load(motor_hurst_dir + animal + '_hurst_concat.npy')
    motor_hurst_avg = [value[0] for value in motor_hurst]
    
    motor_dispen = np.load(motor_dispen_dir + animal + '_dispen.npy')
    motor_gamma = np.load(motor_gamma_dir + animal + '_power.npy')
    motor_theta = np.load(motor_theta_dir + animal + '_power.npy')
    
    #somatosensory 
    soma_hfd = np.load(soma_hfd_dir + animal + '_hfd_concat.npy')
    soma_hfd_avg = [value[0] for value in soma_hfd]
    soma_hurst = np.load(soma_hurst_dir + animal + '_hurst_concat.npy')
    soma_hurst_avg = [value[0] for value in soma_hurst]
    
    soma_dispen = np.load(soma_dispen_dir + animal + '_dispen.npy')
    soma_gamma = np.load(soma_gamma_dir + animal + '_power.npy')
    soma_theta = np.load(soma_theta_dir + animal + '_power.npy')
    
    #somatosensory 
    vis_hfd = np.load(vis_hfd_dir + animal + '_hfd_concat.npy')
    vis_hfd_avg = [value[0] for value in vis_hfd]
    vis_hurst = np.load(vis_hurst_dir + animal + '_hurst_concat.npy')
    vis_hurst_avg = [value[0] for value in vis_hurst]
    
    vis_dispen = np.load(vis_dispen_dir + animal + '_dispen.npy')
    vis_gamma = np.load(vis_gamma_dir + animal + '_power.npy') 
    vis_theta = np.load(vis_theta_dir + animal + '_power.npy') 
    
    #cross cor
    mot_cross_corr_left = np.load(mot_cross_cor_dir + str(animal) + '_mot_left_cross_cor.npy')
    mot_cross_corr_right = np.load(mot_cross_cor_dir + str(animal) + '_mot_right_cross_cor.npy')
    som_cross_corr_left = np.load(som_cross_cor_dir + str(animal) + '_som_left_cross_cor.npy')
    som_cross_corr_right = np.load(som_cross_cor_dir + str(animal) + '_som_right_cross_cor.npy')
    vis_cross_corr_left = np.load(vis_cross_cor_dir + str(animal) + '_vis_left_cross_cor.npy')
    vis_cross_corr_right = np.load(vis_cross_cor_dir + str(animal) + '_vis_right_cross_cor.npy')
    
    #phase lock 
    mot_phase_lock_left = np.load(mot_phase_lock_dir + str(animal) + '_mot_left_phase_lock.npy')
    mot_phase_lock_right = np.load(mot_phase_lock_dir + str(animal) + '_mot_right_phase_lock.npy')
    som_phase_lock_left = np.load(som_phase_lock_dir + str(animal) + '_som_left_phase_lock.npy')
    som_phase_lock_right = np.load(som_phase_lock_dir + str(animal) + '_som_right_phase_lock.npy')
    vis_phase_lock_left = np.load(vis_phase_lock_dir + str(animal) + '_vis_left_phase_lock.npy')
    vis_phase_lock_right = np.load(vis_phase_lock_dir + str(animal) + '_vis_right_phase_lock.npy')
    
    #cross_corr_errors
    error_1 = np.load(cross_cor_dir + animal + '_error_br_1.npy')
    
        
    if len(error_1) > 0:
        br_state = np.delete(br_state, error_1)
        motor_hfd_avg = np.delete(motor_hfd_avg, error_1)
        motor_hurst_avg = np.delete(motor_hurst_avg, error_1)
        motor_dispen = np.delete(motor_dispen, error_1)
        motor_gamma = np.delete(motor_gamma, error_1)
        motor_theta = np.delete(motor_theta, error_1)
        soma_hfd_avg = np.delete(soma_hfd_avg, error_1)
        soma_hurst_avg = np.delete(soma_hurst_avg, error_1)
        soma_dispen = np.delete(soma_dispen, error_1)
        soma_gamma = np.delete(soma_gamma, error_1)
        soma_theta = np.delete(soma_theta, error_1)
        vis_hfd_avg = np.delete(vis_hfd_avg, error_1)
        vis_hurst_avg = np.delete(vis_hurst_avg, error_1)
        vis_dispen = np.delete(vis_dispen, error_1)
        vis_gamma = np.delete(vis_gamma, error_1)
        vis_theta = np.delete(vis_theta, error_1)
        mot_phase_lock_left = np.delete(mot_phase_lock_left, error_1)
        mot_phase_lock_right = np.delete(mot_phase_lock_right, error_1)
        som_phase_lock_left = np.delete(som_phase_lock_left, error_1)
        som_phase_lock_right = np.delete(som_phase_lock_right, error_1)
        vis_phase_lock_left = np.delete(vis_phase_lock_left, error_1)
        vis_phase_lock_right = np.delete(vis_phase_lock_right, error_1)
    else:
        pass
    
    
    #print(len(br_state))
    #print(len(motor_hfd_avg))
    #print(len(motor_hurst_avg))
    #print(len(motor_dispen))
    #print(len(motor_gamma))
    #print(len(soma_hfd_avg))
    #print(len(soma_hurst_avg))
    #print(len(soma_dispen))
    #print(len(soma_gamma))
    #print(len(vis_phase_lock_right))
    
     #clean arrays
    #clean_offset = np.delete(fooof_offset_nan, nan_indices)
    #clean_exponent = np.delete(fooof_exponent_nan, nan_indices)
    #clean_dispen = np.delete(dispen, nan_indices)
    #clean_gamma = np.delete(gamma, nan_indices)
    #clean_br_state = np.delete(br_state, nan_indices)
    
    
    if animal in SYNGAP_het:
        genotype = 1
    elif animal in SYNGAP_wt:
        genotype = 0
        

    region_dict = {'Genotype': [genotype]*len(motor_dispen), 'SleepStage': br_state,
                   'Motor_DispEn': motor_dispen, 'Motor_Hurst': motor_hurst_avg, 
                   'Motor_HFD': motor_hfd_avg,'Motor_Gamma': motor_gamma, 
                   'Motor_Theta': motor_theta,
                   'Soma_DispEn': soma_dispen,'Soma_Hurst': soma_hurst_avg,
                   'Soma_HFD': soma_hfd_avg,'Soma_Gamma': soma_gamma,
                   'Soma_Theta': soma_theta,
                   'Visual_DispEn': vis_dispen,'Visual_Hurst': vis_hurst_avg,
                   'Visual_HFD': vis_hfd_avg, 'Vis_Gamma': vis_gamma, 
                   'Vis_Theta': vis_theta,
                   'Mot_CC_Right': mot_cross_corr_right, 'Mot_CC_Left': mot_cross_corr_left,
                   'Som_CC_Right': som_cross_corr_right, 'Som_CC_Left': som_cross_corr_left,
                   'Vis_CC_Right': vis_cross_corr_right, 'Mot_CC_Left': vis_cross_corr_left,
                   'Mot_PL_Right': mot_phase_lock_right, 'Mot_PL_Left': mot_phase_lock_left,
                   'Som_PL_Right': som_phase_lock_right, 'Som_PL_Left': som_phase_lock_left,
                   'Vis_PL_Right': vis_phase_lock_right, 'Vis_PL_Left': vis_phase_lock_left}
    
    
    region_df = pd.DataFrame(data = region_dict)
    clean_df = region_df[region_df["SleepStage"].isin([0,1,2])]
    print(clean_df)
    feature_df_1_ids.append(clean_df)    

feature_df_2_ids_validation = []
for animal in val_2_ids:
    print(animal)
    #load br file 
    br_1 = pd.read_pickle(br_directory + str(animal) + '_BL1.pkl')
    br_2 = pd.read_pickle(br_directory + str(animal) + '_BL2.pkl')
    br_state_1 = br_1['brainstate'].to_numpy()
    br_state_2 = br_2['brainstate'].to_numpy()
    br_state = np.concatenate([br_state_1, br_state_2])
    
    #motor 
    motor_hfd = np.load(motor_hfd_dir + animal + '_hfd_concat.npy')
    motor_hfd_avg = [value[0] for value in motor_hfd]
    motor_hurst = np.load(motor_hurst_dir + animal + '_hurst_concat.npy')
    motor_hurst_avg = [value[0] for value in motor_hurst]
    
    motor_dispen = np.load(motor_dispen_dir + animal + '_dispen.npy')
    motor_gamma = np.load(motor_gamma_dir + animal + '_power.npy')
    motor_theta = np.load(motor_theta_dir + animal + '_power.npy')
    
    #somatosensory 
    soma_hfd = np.load(soma_hfd_dir + animal + '_hfd_concat.npy')
    soma_hfd_avg = [value[0] for value in soma_hfd]
    soma_hurst = np.load(soma_hurst_dir + animal + '_hurst_concat.npy')
    soma_hurst_avg = [value[0] for value in soma_hurst]
    
    soma_dispen = np.load(soma_dispen_dir + animal + '_dispen.npy')
    soma_gamma = np.load(soma_gamma_dir + animal + '_power.npy') 
    soma_theta = np.load(soma_theta_dir + animal + '_power.npy') 
    
    #somatosensory 
    vis_hfd = np.load(vis_hfd_dir + animal + '_hfd_concat.npy')
    vis_hfd_avg = [value[0] for value in vis_hfd]
    vis_hurst = np.load(vis_hurst_dir + animal + '_hurst_concat.npy')
    vis_hurst_avg = [value[0] for value in vis_hurst]
    
    vis_dispen = np.load(vis_dispen_dir + animal + '_dispen.npy')
    vis_gamma = np.load(vis_gamma_dir + animal + '_power.npy') 
    vis_theta = np.load(vis_theta_dir + animal + '_power.npy') 
    
    #cross cor
    mot_cross_corr_left = np.load(mot_cross_cor_dir + str(animal) + '_mot_left_cross_cor.npy')
    mot_cross_corr_right = np.load(mot_cross_cor_dir + str(animal) + '_mot_right_cross_cor.npy')
    som_cross_corr_left = np.load(som_cross_cor_dir + str(animal) + '_som_left_cross_cor.npy')
    som_cross_corr_right = np.load(som_cross_cor_dir + str(animal) + '_som_right_cross_cor.npy')
    vis_cross_corr_left = np.load(vis_cross_cor_dir + str(animal) + '_vis_left_cross_cor.npy')
    vis_cross_corr_right = np.load(vis_cross_cor_dir + str(animal) + '_vis_right_cross_cor.npy')
    
    #phase lock 
    mot_phase_lock_left = np.load(mot_phase_lock_dir + str(animal) + '_mot_left_phase_lock.npy')
    mot_phase_lock_right = np.load(mot_phase_lock_dir + str(animal) + '_mot_right_phase_lock.npy')
    som_phase_lock_left = np.load(som_phase_lock_dir + str(animal) + '_som_left_phase_lock.npy')
    som_phase_lock_right = np.load(som_phase_lock_dir + str(animal) + '_som_right_phase_lock.npy')
    vis_phase_lock_left = np.load(vis_phase_lock_dir + str(animal) + '_vis_left_phase_lock.npy')
    vis_phase_lock_right = np.load(vis_phase_lock_dir + str(animal) + '_vis_right_phase_lock.npy')
    
    #cross_corr_errors
    error_1 = np.load(cross_cor_dir + animal + '_error_br_1.npy')
    error_2 = np.load(cross_cor_dir + animal + '_error_br_2.npy')
    
    
    if len(error_1) > 0 and len(error_2) >0:
        print(animal + ' error')
        error_2_correct = error_2 + 17280
        errors = np.concatenate([error_1, error_2_correct])
        br_state = np.delete(br_state, errors)
        motor_hfd_avg = np.delete(motor_hfd_avg, errors)
        motor_hurst_avg = np.delete(motor_hurst_avg, errors)
        motor_dispen = np.delete(motor_dispen, errors)
        motor_gamma = np.delete(motor_gamma, errors)
        motor_theta = np.delete(motor_theta, errors)
        soma_hfd_avg = np.delete(soma_hfd_avg, errors)
        soma_hurst_avg = np.delete(soma_hurst_avg, errors)
        soma_dispen = np.delete(soma_dispen, errors)
        soma_gamma = np.delete(soma_gamma, errors)
        soma_theta = np.delete(soma_theta, errors)
        vis_hfd_avg = np.delete(vis_hfd_avg, errors)
        vis_hurst_avg = np.delete(vis_hurst_avg, errors)
        vis_dispen = np.delete(vis_dispen, errors)
        vis_gamma = np.delete(vis_gamma, errors)
        vis_theta = np.delete(vis_theta, errors)
        mot_phase_lock_left = np.delete(mot_phase_lock_left, errors)
        mot_phase_lock_right = np.delete(mot_phase_lock_right, errors)
        som_phase_lock_left = np.delete(som_phase_lock_left, errors)
        som_phase_lock_right = np.delete(som_phase_lock_right, errors)
        vis_phase_lock_left = np.delete(vis_phase_lock_left, errors)
        vis_phase_lock_right = np.delete(vis_phase_lock_right, errors)
        
    elif len(error_1) > 0:
        br_state = np.delete(br_state, error_1)
        motor_hfd_avg = np.delete(motor_hfd_avg, error_1)
        motor_hurst_avg = np.delete(motor_hurst_avg, error_1)
        motor_dispen = np.delete(motor_dispen, error_1)
        motor_gamma = np.delete(motor_gamma, error_1)
        motor_theta = np.delete(motor_theta, error_1)
        soma_hfd_avg = np.delete(soma_hfd_avg, error_1)
        soma_hurst_avg = np.delete(soma_hurst_avg, error_1)
        soma_dispen = np.delete(soma_dispen, error_1)
        soma_gamma = np.delete(soma_gamma, error_1)
        soma_theta = np.delete(soma_theta, error_1)
        vis_hfd_avg = np.delete(vis_hfd_avg, error_1)
        vis_hurst_avg = np.delete(vis_hurst_avg, error_1)
        vis_dispen = np.delete(vis_dispen, error_1)
        vis_gamma = np.delete(vis_gamma, error_1)
        vis_theta = np.delete(vis_theta, error_1)
        mot_phase_lock_left = np.delete(mot_phase_lock_left, error_1)
        mot_phase_lock_right = np.delete(mot_phase_lock_right, error_1)
        som_phase_lock_left = np.delete(som_phase_lock_left, error_1)
        som_phase_lock_right = np.delete(som_phase_lock_right, error_1)
        vis_phase_lock_left = np.delete(vis_phase_lock_left, error_1)
        vis_phase_lock_right = np.delete(vis_phase_lock_right, error_1)
    elif len(error_2) > 0:
        print(animal + ' error 2')
        error_2_br_2 = error_2 + 17280
        br_state = np.delete(br_state, error_2_br_2)
        motor_hfd_avg = np.delete(motor_hfd_avg, error_2_br_2)
        motor_hurst_avg = np.delete(motor_hurst_avg, error_2_br_2)
        motor_dispen = np.delete(motor_dispen, error_2_br_2)
        motor_gamma = np.delete(motor_gamma, error_2_br_2)
        motor_theta = np.delete(motor_theta, error_2_br_2)
        soma_hfd_avg = np.delete(soma_hfd_avg, error_2_br_2)
        soma_hurst_avg = np.delete(soma_hurst_avg, error_2_br_2)
        soma_dispen = np.delete(soma_dispen, error_2_br_2)
        soma_gamma = np.delete(soma_gamma, error_2_br_2)
        soma_theta = np.delete(soma_theta, error_2_br_2)
        vis_hfd_avg = np.delete(vis_hfd_avg, error_2_br_2)
        vis_hurst_avg = np.delete(vis_hurst_avg, error_2_br_2)
        vis_dispen = np.delete(vis_dispen, error_2_br_2)
        vis_gamma = np.delete(vis_gamma, error_2_br_2)
        vis_theta = np.delete(vis_theta, error_2_br_2)
        mot_phase_lock_left = np.delete(mot_phase_lock_left, error_2_br_2)
        mot_phase_lock_right = np.delete(mot_phase_lock_right, error_2_br_2)
        som_phase_lock_left = np.delete(som_phase_lock_left, error_2_br_2)
        som_phase_lock_right = np.delete(som_phase_lock_right, error_2_br_2)
        vis_phase_lock_left = np.delete(vis_phase_lock_left, error_2_br_2)
        vis_phase_lock_right = np.delete(vis_phase_lock_right, error_2_br_2)
    else:
        pass
    
    
    
     #clean arrays
    #clean_offset = np.delete(fooof_offset_nan, nan_indices)
    #clean_exponent = np.delete(fooof_exponent_nan, nan_indices)
    #clean_dispen = np.delete(dispen, nan_indices)
    #clean_gamma = np.delete(gamma, nan_indices)
    #clean_br_state = np.delete(br_state, nan_indices)
    
    
    if animal in SYNGAP_het:
        genotype = 1
    elif animal in SYNGAP_wt:
        genotype = 0
        

    region_dict = {'Genotype': [genotype]*len(motor_dispen), 'SleepStage': br_state,
                   'Motor_DispEn': motor_dispen, 'Motor_Hurst': motor_hurst_avg, 
                   'Motor_HFD': motor_hfd_avg,'Motor_Gamma': motor_gamma, 
                   'Motor_Theta': motor_theta,
                   'Soma_DispEn': soma_dispen,'Soma_Hurst': soma_hurst_avg,
                   'Soma_HFD': soma_hfd_avg,'Soma_Gamma': soma_gamma,
                   'Soma_Theta': soma_theta,
                   'Visual_DispEn': vis_dispen,'Visual_Hurst': vis_hurst_avg,
                   'Visual_HFD': vis_hfd_avg, 'Vis_Gamma': vis_gamma, 
                   'Soma_Theta': soma_theta,
                   'Mot_CC_Right': mot_cross_corr_right, 'Mot_CC_Left': mot_cross_corr_left,
                   'Som_CC_Right': som_cross_corr_right, 'Som_CC_Left': som_cross_corr_left,
                   'Vis_CC_Right': vis_cross_corr_right, 'Mot_CC_Left': vis_cross_corr_left,
                   'Mot_PL_Right': mot_phase_lock_right, 'Mot_PL_Left': mot_phase_lock_left,
                   'Som_PL_Right': som_phase_lock_right, 'Som_PL_Left': som_phase_lock_left,
                   'Vis_PL_Right': vis_phase_lock_right, 'Vis_PL_Left': vis_phase_lock_left}
    
    
    region_df = pd.DataFrame(data = region_dict)
    clean_df = region_df[region_df["SleepStage"].isin([0,1,2])]
    print(len(clean_df))
    print(clean_df)
    feature_df_2_ids_validation.append(clean_df)
    
feature_df_1_ids_validation = []
for animal in val_1_ids:
    print(animal)
    #load br file 
    br_1 = pd.read_pickle(br_directory + str(animal) + '_BL1.pkl')
    br_state = br_1['brainstate'].to_numpy()
    
    #motor 
    motor_hfd = np.load(motor_hfd_dir + animal + '_hfd_concat.npy')
    motor_hfd_avg = [value[0] for value in motor_hfd]
    motor_hurst = np.load(motor_hurst_dir + animal + '_hurst_concat.npy')
    motor_hurst_avg = [value[0] for value in motor_hurst]
    
    motor_dispen = np.load(motor_dispen_dir + animal + '_dispen.npy')
    motor_gamma = np.load(motor_gamma_dir + animal + '_power.npy') 
    motor_theta = np.load(motor_theta_dir + animal + '_power.npy') 
    
    #somatosensory 
    soma_hfd = np.load(soma_hfd_dir + animal + '_hfd_concat.npy')
    soma_hfd_avg = [value[0] for value in soma_hfd]
    soma_hurst = np.load(soma_hurst_dir + animal + '_hurst_concat.npy')
    soma_hurst_avg = [value[0] for value in soma_hurst]
    
    soma_dispen = np.load(soma_dispen_dir + animal + '_dispen.npy')
    soma_gamma = np.load(soma_gamma_dir + animal + '_power.npy') 
    soma_theta = np.load(soma_theta_dir + animal + '_power.npy') 
    
    #somatosensory 
    vis_hfd = np.load(vis_hfd_dir + animal + '_hfd_concat.npy')
    vis_hfd_avg = [value[0] for value in vis_hfd]
    vis_hurst = np.load(vis_hurst_dir + animal + '_hurst_concat.npy')
    vis_hurst_avg = [value[0] for value in vis_hurst]
    
    vis_dispen = np.load(vis_dispen_dir + animal + '_dispen.npy')
    vis_gamma = np.load(vis_gamma_dir + animal + '_power.npy') 
    vis_theta = np.load(vis_theta_dir + animal + '_power.npy')
    
    #cross cor
    mot_cross_corr_left = np.load(mot_cross_cor_dir + str(animal) + '_mot_left_cross_cor.npy')
    mot_cross_corr_right = np.load(mot_cross_cor_dir + str(animal) + '_mot_right_cross_cor.npy')
    som_cross_corr_left = np.load(som_cross_cor_dir + str(animal) + '_som_left_cross_cor.npy')
    som_cross_corr_right = np.load(som_cross_cor_dir + str(animal) + '_som_right_cross_cor.npy')
    vis_cross_corr_left = np.load(vis_cross_cor_dir + str(animal) + '_vis_left_cross_cor.npy')
    vis_cross_corr_right = np.load(vis_cross_cor_dir + str(animal) + '_vis_right_cross_cor.npy')
    
    #phase lock 
    mot_phase_lock_left = np.load(mot_phase_lock_dir + str(animal) + '_mot_left_phase_lock.npy')
    mot_phase_lock_right = np.load(mot_phase_lock_dir + str(animal) + '_mot_right_phase_lock.npy')
    som_phase_lock_left = np.load(som_phase_lock_dir + str(animal) + '_som_left_phase_lock.npy')
    som_phase_lock_right = np.load(som_phase_lock_dir + str(animal) + '_som_right_phase_lock.npy')
    vis_phase_lock_left = np.load(vis_phase_lock_dir + str(animal) + '_vis_left_phase_lock.npy')
    vis_phase_lock_right = np.load(vis_phase_lock_dir + str(animal) + '_vis_right_phase_lock.npy')
    
    #cross_corr_errors
    error_1 = np.load(cross_cor_dir + animal + '_error_br_1.npy')
    
        
    if len(error_1) > 0:
        br_state = np.delete(br_state, error_1)
        motor_hfd_avg = np.delete(motor_hfd_avg, error_1)
        motor_hurst_avg = np.delete(motor_hurst_avg, error_1)
        motor_dispen = np.delete(motor_dispen, error_1)
        motor_gamma = np.delete(motor_gamma, error_1)
        motor_theta = np.delete(motor_theta, error_1)
        soma_hfd_avg = np.delete(soma_hfd_avg, error_1)
        soma_hurst_avg = np.delete(soma_hurst_avg, error_1)
        soma_dispen = np.delete(soma_dispen, error_1)
        soma_gamma = np.delete(soma_gamma, error_1)
        soma_theta = np.delete(soma_theta, error_1)
        vis_hfd_avg = np.delete(vis_hfd_avg, error_1)
        vis_hurst_avg = np.delete(vis_hurst_avg, error_1)
        vis_dispen = np.delete(vis_dispen, error_1)
        vis_gamma = np.delete(vis_gamma, error_1)
        vis_theta = np.delete(vis_theta, error_1)
        mot_phase_lock_left = np.delete(mot_phase_lock_left, error_1)
        mot_phase_lock_right = np.delete(mot_phase_lock_right, error_1)
        som_phase_lock_left = np.delete(som_phase_lock_left, error_1)
        som_phase_lock_right = np.delete(som_phase_lock_right, error_1)
        vis_phase_lock_left = np.delete(vis_phase_lock_left, error_1)
        vis_phase_lock_right = np.delete(vis_phase_lock_right, error_1)
    else:
        pass
    
    
    #print(len(br_state))
    #print(len(motor_hfd_avg))
    #print(len(motor_hurst_avg))
    #print(len(motor_dispen))
    #print(len(motor_gamma))
    #print(len(soma_hfd_avg))
    #print(len(soma_hurst_avg))
    #print(len(soma_dispen))
    #print(len(soma_gamma))
    #print(len(vis_phase_lock_right))
    
     #clean arrays
    #clean_offset = np.delete(fooof_offset_nan, nan_indices)
    #clean_exponent = np.delete(fooof_exponent_nan, nan_indices)
    #clean_dispen = np.delete(dispen, nan_indices)
    #clean_gamma = np.delete(gamma, nan_indices)
    #clean_br_state = np.delete(br_state, nan_indices)
    
    
    if animal in SYNGAP_het:
        genotype = 1
    elif animal in SYNGAP_wt:
        genotype = 0
        

    region_dict = {'Genotype': [genotype]*len(motor_dispen), 'SleepStage': br_state,
                   'Motor_DispEn': motor_dispen, 'Motor_Hurst': motor_hurst_avg, 
                   'Motor_HFD': motor_hfd_avg,'Motor_Gamma': motor_gamma, 
                   'Motor_Theta': motor_theta,
                   'Soma_DispEn': soma_dispen,'Soma_Hurst': soma_hurst_avg,
                   'Soma_HFD': soma_hfd_avg,'Soma_Gamma': soma_gamma,
                   'Soma_Theta': soma_theta,
                   'Visual_DispEn': vis_dispen,'Visual_Hurst': vis_hurst_avg,
                   'Visual_HFD': vis_hfd_avg, 'Vis_Gamma': vis_gamma, 
                   'Vis_Theta': vis_theta,
                   'Mot_CC_Right': mot_cross_corr_right, 'Mot_CC_Left': mot_cross_corr_left,
                   'Som_CC_Right': som_cross_corr_right, 'Som_CC_Left': som_cross_corr_left,
                   'Vis_CC_Right': vis_cross_corr_right, 'Mot_CC_Left': vis_cross_corr_left,
                   'Mot_PL_Right': mot_phase_lock_right, 'Mot_PL_Left': mot_phase_lock_left,
                   'Som_PL_Right': som_phase_lock_right, 'Som_PL_Left': som_phase_lock_left,
                   'Vis_PL_Right': vis_phase_lock_right, 'Vis_PL_Left': vis_phase_lock_left}
    
    
    region_df = pd.DataFrame(data = region_dict)
    clean_df = region_df[region_df["SleepStage"].isin([0,1,2])]
    print(clean_df)
    feature_df_1_ids_validation.append(clean_df)
    
feature_df_test_2_ids = []
for animal in test_2_ids:
    print(animal)
    #load br file 
    br_1 = pd.read_pickle(br_directory + str(animal) + '_BL1.pkl')
    br_2 = pd.read_pickle(br_directory + str(animal) + '_BL2.pkl')
    br_state_1 = br_1['brainstate'].to_numpy()
    br_state_2 = br_2['brainstate'].to_numpy()
    br_state = np.concatenate([br_state_1, br_state_2])
    
    #motor 
    motor_hfd = np.load(motor_hfd_dir + animal + '_hfd_concat.npy')
    motor_hfd_avg = [value[0] for value in motor_hfd]
    motor_hurst = np.load(motor_hurst_dir + animal + '_hurst_concat.npy')
    motor_hurst_avg = [value[0] for value in motor_hurst]
    
    motor_dispen = np.load(motor_dispen_dir + animal + '_dispen.npy')
    motor_gamma = np.load(motor_gamma_dir + animal + '_power.npy') 
    motor_theta = np.load(motor_theta_dir + animal + '_power.npy') 
    
    #somatosensory 
    soma_hfd = np.load(soma_hfd_dir + animal + '_hfd_concat.npy')
    soma_hfd_avg = [value[0] for value in soma_hfd]
    soma_hurst = np.load(soma_hurst_dir + animal + '_hurst_concat.npy')
    soma_hurst_avg = [value[0] for value in soma_hurst]
    
    soma_dispen = np.load(soma_dispen_dir + animal + '_dispen.npy')
    soma_gamma = np.load(soma_gamma_dir + animal + '_power.npy') 
    soma_theta = np.load(soma_theta_dir + animal + '_power.npy') 
    
    #somatosensory 
    vis_hfd = np.load(vis_hfd_dir + animal + '_hfd_concat.npy')
    vis_hfd_avg = [value[0] for value in vis_hfd]
    vis_hurst = np.load(vis_hurst_dir + animal + '_hurst_concat.npy')
    vis_hurst_avg = [value[0] for value in vis_hurst]
    
    vis_dispen = np.load(vis_dispen_dir + animal + '_dispen.npy')
    vis_gamma = np.load(vis_gamma_dir + animal + '_power.npy') 
    vis_theta = np.load(vis_theta_dir + animal + '_power.npy') 
    
    #cross cor
    mot_cross_corr_left = np.load(mot_cross_cor_dir + str(animal) + '_mot_left_cross_cor.npy')
    mot_cross_corr_right = np.load(mot_cross_cor_dir + str(animal) + '_mot_right_cross_cor.npy')
    som_cross_corr_left = np.load(som_cross_cor_dir + str(animal) + '_som_left_cross_cor.npy')
    som_cross_corr_right = np.load(som_cross_cor_dir + str(animal) + '_som_right_cross_cor.npy')
    vis_cross_corr_left = np.load(vis_cross_cor_dir + str(animal) + '_vis_left_cross_cor.npy')
    vis_cross_corr_right = np.load(vis_cross_cor_dir + str(animal) + '_vis_right_cross_cor.npy')
    
    #phase lock 
    mot_phase_lock_left = np.load(mot_phase_lock_dir + str(animal) + '_mot_left_phase_lock.npy')
    mot_phase_lock_right = np.load(mot_phase_lock_dir + str(animal) + '_mot_right_phase_lock.npy')
    som_phase_lock_left = np.load(som_phase_lock_dir + str(animal) + '_som_left_phase_lock.npy')
    som_phase_lock_right = np.load(som_phase_lock_dir + str(animal) + '_som_right_phase_lock.npy')
    vis_phase_lock_left = np.load(vis_phase_lock_dir + str(animal) + '_vis_left_phase_lock.npy')
    vis_phase_lock_right = np.load(vis_phase_lock_dir + str(animal) + '_vis_right_phase_lock.npy')
    
    #cross_corr_errors
    error_1 = np.load(cross_cor_dir + animal + '_error_br_1.npy')
    error_2 = np.load(cross_cor_dir + animal + '_error_br_2.npy')
    
    
    if len(error_1) > 0 and len(error_2) >0:
        print(animal + ' error')
        error_2_correct = error_2 + 17280
        errors = np.concatenate([error_1, error_2_correct])
        br_state = np.delete(br_state, errors)
        motor_hfd_avg = np.delete(motor_hfd_avg, errors)
        motor_hurst_avg = np.delete(motor_hurst_avg, errors)
        motor_dispen = np.delete(motor_dispen, errors)
        motor_gamma = np.delete(motor_gamma, errors)
        soma_hfd_avg = np.delete(soma_hfd_avg, errors)
        soma_hurst_avg = np.delete(soma_hurst_avg, errors)
        soma_dispen = np.delete(soma_dispen, errors)
        soma_gamma = np.delete(soma_gamma, errors)
        vis_hfd_avg = np.delete(vis_hfd_avg, errors)
        vis_hurst_avg = np.delete(vis_hurst_avg, errors)
        vis_dispen = np.delete(vis_dispen, errors)
        vis_gamma = np.delete(vis_gamma, errors)
        mot_phase_lock_left = np.delete(mot_phase_lock_left, errors)
        mot_phase_lock_right = np.delete(mot_phase_lock_right, errors)
        som_phase_lock_left = np.delete(som_phase_lock_left, errors)
        som_phase_lock_right = np.delete(som_phase_lock_right, errors)
        vis_phase_lock_left = np.delete(vis_phase_lock_left, errors)
        vis_phase_lock_right = np.delete(vis_phase_lock_right, errors)
        
    elif len(error_1) > 0:
        br_state = np.delete(br_state, error_1)
        motor_hfd_avg = np.delete(motor_hfd_avg, error_1)
        motor_hurst_avg = np.delete(motor_hurst_avg, error_1)
        motor_dispen = np.delete(motor_dispen, error_1)
        motor_gamma = np.delete(motor_gamma, error_1)
        soma_hfd_avg = np.delete(soma_hfd_avg, error_1)
        soma_hurst_avg = np.delete(soma_hurst_avg, error_1)
        soma_dispen = np.delete(soma_dispen, error_1)
        soma_gamma = np.delete(soma_gamma, error_1)
        vis_hfd_avg = np.delete(vis_hfd_avg, error_1)
        vis_hurst_avg = np.delete(vis_hurst_avg, error_1)
        vis_dispen = np.delete(vis_dispen, error_1)
        vis_gamma = np.delete(vis_gamma, error_1)
        mot_phase_lock_left = np.delete(mot_phase_lock_left, error_1)
        mot_phase_lock_right = np.delete(mot_phase_lock_right, error_1)
        som_phase_lock_left = np.delete(som_phase_lock_left, error_1)
        som_phase_lock_right = np.delete(som_phase_lock_right, error_1)
        vis_phase_lock_left = np.delete(vis_phase_lock_left, error_1)
        vis_phase_lock_right = np.delete(vis_phase_lock_right, error_1)
    elif len(error_2) > 0:
        print(animal + ' error 2')
        error_2_br_2 = error_2 + 17280
        br_state = np.delete(br_state, error_2_br_2)
        motor_hfd_avg = np.delete(motor_hfd_avg, error_2_br_2)
        motor_hurst_avg = np.delete(motor_hurst_avg, error_2_br_2)
        motor_dispen = np.delete(motor_dispen, error_2_br_2)
        motor_gamma = np.delete(motor_gamma, error_2_br_2)
        soma_hfd_avg = np.delete(soma_hfd_avg, error_2_br_2)
        soma_hurst_avg = np.delete(soma_hurst_avg, error_2_br_2)
        soma_dispen = np.delete(soma_dispen, error_2_br_2)
        soma_gamma = np.delete(soma_gamma, error_2_br_2)
        vis_hfd_avg = np.delete(vis_hfd_avg, error_2_br_2)
        vis_hurst_avg = np.delete(vis_hurst_avg, error_2_br_2)
        vis_dispen = np.delete(vis_dispen, error_2_br_2)
        vis_gamma = np.delete(vis_gamma, error_2_br_2)
        mot_phase_lock_left = np.delete(mot_phase_lock_left, error_2_br_2)
        mot_phase_lock_right = np.delete(mot_phase_lock_right, error_2_br_2)
        som_phase_lock_left = np.delete(som_phase_lock_left, error_2_br_2)
        som_phase_lock_right = np.delete(som_phase_lock_right, error_2_br_2)
        vis_phase_lock_left = np.delete(vis_phase_lock_left, error_2_br_2)
        vis_phase_lock_right = np.delete(vis_phase_lock_right, error_2_br_2)
    else:
        pass
    
    
    
     #clean arrays
    #clean_offset = np.delete(fooof_offset_nan, nan_indices)
    #clean_exponent = np.delete(fooof_exponent_nan, nan_indices)
    #clean_dispen = np.delete(dispen, nan_indices)
    #clean_gamma = np.delete(gamma, nan_indices)
    #clean_br_state = np.delete(br_state, nan_indices)
    
    
    if animal in SYNGAP_het:
        genotype = 1
    elif animal in SYNGAP_wt:
        genotype = 0
        

    region_dict = {'Genotype': [genotype]*len(motor_dispen), 'SleepStage': br_state,
                   'Motor_DispEn': motor_dispen, 'Motor_Hurst': motor_hurst_avg, 
                   'Motor_HFD': motor_hfd_avg,'Motor_Gamma': motor_gamma, 
                   'Motor_Theta': motor_theta,
                   'Soma_DispEn': soma_dispen,'Soma_Hurst': soma_hurst_avg,
                   'Soma_HFD': soma_hfd_avg,'Soma_Gamma': soma_gamma,
                   'Soma_Theta': soma_theta,
                   'Visual_DispEn': vis_dispen,'Visual_Hurst': vis_hurst_avg,
                   'Visual_HFD': vis_hfd_avg, 'Vis_Gamma': vis_gamma, 
                   'Vis_Theta': vis_theta,
                   'Mot_CC_Right': mot_cross_corr_right, 'Mot_CC_Left': mot_cross_corr_left,
                   'Som_CC_Right': som_cross_corr_right, 'Som_CC_Left': som_cross_corr_left,
                   'Vis_CC_Right': vis_cross_corr_right, 'Mot_CC_Left': vis_cross_corr_left,
                   'Mot_PL_Right': mot_phase_lock_right, 'Mot_PL_Left': mot_phase_lock_left,
                   'Som_PL_Right': som_phase_lock_right, 'Som_PL_Left': som_phase_lock_left,
                   'Vis_PL_Right': vis_phase_lock_right, 'Vis_PL_Left': vis_phase_lock_left}
    
    
    region_df = pd.DataFrame(data = region_dict)
    clean_df = region_df[region_df["SleepStage"].isin([0,1,2])]
    print(len(clean_df))
    print(clean_df)
    feature_df_test_2_ids.append(clean_df)
    

feature_df_1_test_ids = []
for animal in test_1_ids:
    print(animal)
    #load br file 
    br_1 = pd.read_pickle(br_directory + str(animal) + '_BL1.pkl')
    br_state = br_1['brainstate'].to_numpy()
    
    #motor 
    motor_hfd = np.load(motor_hfd_dir + animal + '_hfd_concat.npy')
    motor_hfd_avg = [value[0] for value in motor_hfd]
    motor_hurst = np.load(motor_hurst_dir + animal + '_hurst_concat.npy')
    motor_hurst_avg = [value[0] for value in motor_hurst]
    
    motor_dispen = np.load(motor_dispen_dir + animal + '_dispen.npy')
    motor_gamma = np.load(motor_gamma_dir + animal + '_power.npy') 
    motor_theta = np.load(motor_theta_dir + animal + '_power.npy') 
    
    #somatosensory 
    soma_hfd = np.load(soma_hfd_dir + animal + '_hfd_concat.npy')
    soma_hfd_avg = [value[0] for value in soma_hfd]
    soma_hurst = np.load(soma_hurst_dir + animal + '_hurst_concat.npy')
    soma_hurst_avg = [value[0] for value in soma_hurst]
    
    soma_dispen = np.load(soma_dispen_dir + animal + '_dispen.npy')
    soma_gamma = np.load(soma_gamma_dir + animal + '_power.npy') 
    soma_theta = np.load(soma_theta_dir + animal + '_power.npy') 
    
    #somatosensory 
    vis_hfd = np.load(vis_hfd_dir + animal + '_hfd_concat.npy')
    vis_hfd_avg = [value[0] for value in vis_hfd]
    vis_hurst = np.load(vis_hurst_dir + animal + '_hurst_concat.npy')
    vis_hurst_avg = [value[0] for value in vis_hurst]
    
    vis_dispen = np.load(vis_dispen_dir + animal + '_dispen.npy')
    vis_gamma = np.load(vis_gamma_dir + animal + '_power.npy') 
    vis_theta = np.load(vis_theta_dir + animal + '_power.npy') 
    
    #cross cor
    mot_cross_corr_left = np.load(mot_cross_cor_dir + str(animal) + '_mot_left_cross_cor.npy')
    mot_cross_corr_right = np.load(mot_cross_cor_dir + str(animal) + '_mot_right_cross_cor.npy')
    som_cross_corr_left = np.load(som_cross_cor_dir + str(animal) + '_som_left_cross_cor.npy')
    som_cross_corr_right = np.load(som_cross_cor_dir + str(animal) + '_som_right_cross_cor.npy')
    vis_cross_corr_left = np.load(vis_cross_cor_dir + str(animal) + '_vis_left_cross_cor.npy')
    vis_cross_corr_right = np.load(vis_cross_cor_dir + str(animal) + '_vis_right_cross_cor.npy')
    
    #phase lock 
    mot_phase_lock_left = np.load(mot_phase_lock_dir + str(animal) + '_mot_left_phase_lock.npy')
    mot_phase_lock_right = np.load(mot_phase_lock_dir + str(animal) + '_mot_right_phase_lock.npy')
    som_phase_lock_left = np.load(som_phase_lock_dir + str(animal) + '_som_left_phase_lock.npy')
    som_phase_lock_right = np.load(som_phase_lock_dir + str(animal) + '_som_right_phase_lock.npy')
    vis_phase_lock_left = np.load(vis_phase_lock_dir + str(animal) + '_vis_left_phase_lock.npy')
    vis_phase_lock_right = np.load(vis_phase_lock_dir + str(animal) + '_vis_right_phase_lock.npy')
    
    #cross_corr_errors
    error_1 = np.load(cross_cor_dir + animal + '_error_br_1.npy')
    
        
    if len(error_1) > 0:
        br_state = np.delete(br_state, error_1)
        motor_hfd_avg = np.delete(motor_hfd_avg, error_1)
        motor_hurst_avg = np.delete(motor_hurst_avg, error_1)
        motor_dispen = np.delete(motor_dispen, error_1)
        motor_gamma = np.delete(motor_gamma, error_1)
        motor_theta = np.delete(motor_theta, error_1)
        soma_hfd_avg = np.delete(soma_hfd_avg, error_1)
        soma_hurst_avg = np.delete(soma_hurst_avg, error_1)
        soma_dispen = np.delete(soma_dispen, error_1)
        soma_gamma = np.delete(soma_gamma, error_1)
        soma_theta = np.delete(soma_theta, error_1)
        vis_hfd_avg = np.delete(vis_hfd_avg, error_1)
        vis_hurst_avg = np.delete(vis_hurst_avg, error_1)
        vis_dispen = np.delete(vis_dispen, error_1)
        vis_gamma = np.delete(vis_gamma, error_1)
        vis_theta = np.delete(vis_theta, error_1)
        mot_phase_lock_left = np.delete(mot_phase_lock_left, error_1)
        mot_phase_lock_right = np.delete(mot_phase_lock_right, error_1)
        som_phase_lock_left = np.delete(som_phase_lock_left, error_1)
        som_phase_lock_right = np.delete(som_phase_lock_right, error_1)
        vis_phase_lock_left = np.delete(vis_phase_lock_left, error_1)
        vis_phase_lock_right = np.delete(vis_phase_lock_right, error_1)
    else:
        pass
    
    
    #print(len(br_state))
    #print(len(motor_hfd_avg))
    #print(len(motor_hurst_avg))
    #print(len(motor_dispen))
    #print(len(motor_gamma))
    #print(len(soma_hfd_avg))
    #print(len(soma_hurst_avg))
    #print(len(soma_dispen))
    #print(len(soma_gamma))
    #print(len(vis_phase_lock_right))
    
     #clean arrays
    #clean_offset = np.delete(fooof_offset_nan, nan_indices)
    #clean_exponent = np.delete(fooof_exponent_nan, nan_indices)
    #clean_dispen = np.delete(dispen, nan_indices)
    #clean_gamma = np.delete(gamma, nan_indices)
    #clean_br_state = np.delete(br_state, nan_indices)
    
    
    if animal in SYNGAP_het:
        genotype = 1
    elif animal in SYNGAP_wt:
        genotype = 0
        

    region_dict = {'Genotype': [genotype]*len(motor_dispen), 'SleepStage': br_state,
                   'Motor_DispEn': motor_dispen, 'Motor_Hurst': motor_hurst_avg, 
                   'Motor_HFD': motor_hfd_avg,'Motor_Gamma': motor_gamma, 
                   'Motor_Theta': motor_theta,
                   'Soma_DispEn': soma_dispen,'Soma_Hurst': soma_hurst_avg,
                   'Soma_HFD': soma_hfd_avg,'Soma_Gamma': soma_gamma,
                   'Soma_Theta': soma_theta,
                   'Visual_DispEn': vis_dispen,'Visual_Hurst': vis_hurst_avg,
                   'Visual_HFD': vis_hfd_avg, 'Vis_Gamma': vis_gamma, 
                   'Vis_Theta': vis_theta,
                   'Mot_CC_Right': mot_cross_corr_right, 'Mot_CC_Left': mot_cross_corr_left,
                   'Som_CC_Right': som_cross_corr_right, 'Som_CC_Left': som_cross_corr_left,
                   'Vis_CC_Right': vis_cross_corr_right, 'Mot_CC_Left': vis_cross_corr_left,
                   'Mot_PL_Right': mot_phase_lock_right, 'Mot_PL_Left': mot_phase_lock_left,
                   'Som_PL_Right': som_phase_lock_right, 'Som_PL_Left': som_phase_lock_left,
                   'Vis_PL_Right': vis_phase_lock_right, 'Vis_PL_Left': vis_phase_lock_left}
    
    
    region_df = pd.DataFrame(data = region_dict)
    clean_df = region_df[region_df["SleepStage"].isin([0,1,2])]
    print(len(clean_df))
    print(clean_df)
    feature_df_1_test_ids.append(clean_df)
    
train_concat_1 = pd.concat(feature_df_1_ids)
train_concat_2 = pd.concat(feature_df_2_ids)
feature_concat_train = pd.concat([train_concat_1, train_concat_2])

val_concat_1 = pd.concat(feature_df_1_ids_validation)
val_concat_2 = pd.concat(feature_df_2_ids_validation)
feature_concat_validation = pd.concat([val_concat_1, val_concat_2])

test_concat_1 = pd.concat(feature_df_test_2_ids)
test_concat_2 = pd.concat(feature_df_1_test_ids)
feature_concat_test = pd.concat([test_concat_1, test_concat_2])

X_train = feature_concat_train.iloc[:, 1:]
y_train = feature_concat_train.iloc[:, 0]
X_val = feature_concat_validation.iloc[:, 1:]
y_val = feature_concat_validation.iloc[:, 0]
X_test = feature_concat_test.iloc[:, 1:]
y_test = feature_concat_test.iloc[:, 0]

print('wildtype train')
print(len(feature_concat_train.loc[feature_concat_train['Genotype'] == 0]))
print('mutant train')
print(len(feature_concat_train.loc[feature_concat_train['Genotype'] == 1]))
print('wildtype val')
print(len(feature_concat_validation.loc[feature_concat_validation['Genotype'] == 0]))
print('mutant val')
print(len(feature_concat_validation.loc[feature_concat_validation['Genotype'] == 1]))
print('mutant test')
print(len(feature_concat_test.loc[feature_concat_test['Genotype'] == 1]))
print('wildtype test')
print(len(feature_concat_test.loc[feature_concat_test['Genotype'] == 0]))

undersample = RandomUnderSampler(sampling_strategy = 'majority')
X_train_new, y_train_new = undersample.fit_resample(X_train, y_train)
X_test_new, y_test_new = undersample.fit_resample(X_test, y_test)

options = {'max_depth': hp.quniform('max_depth', 1, 8, 1), #tree
            'min_child_weight': hp.loguniform('min_child_weight', -2, 3),
            'subsample': hp.uniform('subsample', 0.5, 1), #stochastic
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'reg_alpha': hp.uniform('reg_alpha', 0, 10), 
            'reg_lambda': hp.uniform('reg_lambda', 1, 10),
            'gamma': hp.loguniform('gamma', -10, 10),
            'learning_rate': hp.loguniform('learning_rate', -7, 0), 
            'random_state': 42
          }


trials = Trials()
best = fmin(fn=lambda space: hyperparameter_tuning(space, X_train_new,
                                                       y_train_new, X_val, y_val),
            space = options, algo = tpe.suggest, max_evals = 2000, trials = trials)
print(best)