import os 
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, KFold
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from imblearn.under_sampling import RandomUnderSampler

import xgboost as xgb
import xgbfir
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import dtreeviz
from typing import Any, Dict, Union

from yellowbrick import model_selection as ms
from yellowbrick.model_selection import validation_curve
from hyperopt import fmin, tpe, hp, Trials

from hyperparam_hyperopt import hyperparameter_tuning


ids = ['131', '229','236', '237', '241', '362', '366', 
       '373', '132', '138', '140', '402', '228', '238',
       '363', '367', '378', '129', '137', '239', '383', '364',
       '365', '371', '382', '404', '130', '139', '401', '240',
       '227', '375', '424', '433', '430', '368', '369']

GRIN2B_ID_list = [ '138', '140', '402', '228', '238',
                 '363', '367', '378', '129', '137', '239', '383', '364'] 

GRIN_het_IDs = ['131', '130', '129', '228', '227', '229', '373', '138', '137',
                '139','236', '237', '239', '241', '364', '367', '368', '424',
                '433']
GRIN_wt_IDs = ['132', '362', '363', '375', '378', '382', '383', '140', '238',
                '240', '365', '366', '369', '371', '401', '402', '404', '430']


genotype = []
for animal in ids:
    if animal in GRIN_het_IDs:
        gen = 1
        genotype.append(gen)
    elif animal in GRIN_wt_IDs:
        gen = 0
        genotype.append(gen)

train_test_dict = {'Animal_ID': ids, 'Genotype': genotype}
train_test_df = pd.DataFrame(data = train_test_dict)

#split into train and test 
splitter = GroupShuffleSplit(test_size=.30, n_splits=2, random_state = 7)
split = splitter.split(train_test_df, groups=train_test_df['Animal_ID'])
train_inds, test_inds = next(split)

train = train_test_df.iloc[train_inds]
test = train_test_df.iloc[test_inds]

#split training set into train and validation for hyperparameter tuning
splitter_2 = GroupShuffleSplit(test_size=.30, n_splits=2, random_state = 7)
split_2 = splitter_2.split(train, groups=train['Animal_ID'])
train_inds_2, val_inds = next(split_2)

train_idx = train.iloc[train_inds_2]
val_idx = train.iloc[val_inds]

#final train, val and test indices
final_train_set = train_idx['Animal_ID'].to_list()
final_val_set = val_idx['Animal_ID'].to_list()
final_test_set = test['Animal_ID'].to_list()

motor_direc = '/home/melissa/RESULTS/XGBoost/GRIN2B/Motor/'
soma_direc = '/home/melissa/RESULTS/XGBoost/GRIN2B/Somatosensory/'
visual_direc = '/home/melissa/RESULTS/XGBoost/GRIN2B/Visual/'

motor_hurst_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Motor/Hurst/'
motor_hfd_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Motor/HFD/'
motor_dispen_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Motor/DispEn/'
motor_gamma_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Motor/Gamma_Power/'

soma_hurst_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Somatosensory/Hurst/'
soma_hfd_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Somatosensory/HFD/'
soma_dispen_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Somatosensory/DispEn/'
soma_gamma_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Somatosensory/Gamma_Power/'

vis_hurst_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Visual/Hurst/'
vis_hfd_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Visual/HFD/'
vis_dispen_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Visual/DispEn/'
vis_gamma_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Visual/Gamma_Power/'

#connectivity indices
cross_cor_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/CrossCorr/'
mot_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Motor/CrossCorr_Mot/' 
som_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Somatosensory/CrossCorr_Som/'
vis_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Visual/CrossCorr_Vis/'

mot_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Motor/Phase_Lock_Mot/' 
som_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Somatosensory/Phase_Lock_Som/'
vis_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/GRIN2B/Visual/Phase_Lock_Vis/'

br_directory = '/home/melissa/PREPROCESSING/GRIN2B/GRIN2B_numpy/'


#training features 
feature_df_train = []
for animal in final_train_set:
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
    
    #somatosensory 
    soma_hfd = np.load(soma_hfd_dir + animal + '_hfd_concat.npy')
    soma_hfd_avg = [value[0] for value in soma_hfd]
    soma_hurst = np.load(soma_hurst_dir + animal + '_hurst_concat.npy')
    soma_hurst_avg = [value[0] for value in soma_hurst]
    
    soma_dispen = np.load(soma_dispen_dir + animal + '_dispen.npy')
    soma_gamma = np.load(soma_gamma_dir + animal + '_power.npy') 
    
    #somatosensory 
    vis_hfd = np.load(vis_hfd_dir + animal + '_hfd_concat.npy')
    vis_hfd_avg = [value[0] for value in vis_hfd]
    vis_hurst = np.load(vis_hurst_dir + animal + '_hurst_concat.npy')
    vis_hurst_avg = [value[0] for value in vis_hurst]
    
    vis_dispen = np.load(vis_dispen_dir + animal + '_dispen.npy')
    vis_gamma = np.load(vis_gamma_dir + animal + '_power.npy') 
    
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

    
    
    if animal in GRIN_het_IDs:
        genotype = 1
    elif animal in GRIN_wt_IDs:
        genotype = 0
        

    region_dict = {'Genotype': [genotype]*len(motor_dispen), 'SleepStage': br_state,
                   'Motor_DispEn': motor_dispen, 'Motor_Hurst': motor_hurst_avg, 
                   'Motor_HFD': motor_hfd_avg,'Motor_Gamma': motor_gamma, 
                   'Soma_DispEn': soma_dispen,'Soma_Hurst': soma_hurst_avg,
                   'Soma_HFD': soma_hfd_avg,'Soma_Gamma': soma_gamma,
                   'Visual_DispEn': vis_dispen,'Visual_Hurst': vis_hurst_avg,
                   'Visual_HFD': vis_hfd_avg, 'Vis_Gamma': vis_gamma, 
                   'Mot_CC_Right': mot_cross_corr_right, 'Mot_CC_Left': mot_cross_corr_left,
                   'Som_CC_Right': som_cross_corr_right, 'Som_CC_Left': som_cross_corr_left,
                   'Vis_CC_Right': vis_cross_corr_right, 'Mot_CC_Left': vis_cross_corr_left,
                   'Mot_PL_Right': mot_phase_lock_right, 'Mot_PL_Left': mot_phase_lock_left,
                   'Som_PL_Right': som_phase_lock_right, 'Som_PL_Left': som_phase_lock_left,
                   'Vis_PL_Right': vis_phase_lock_right, 'Vis_PL_Left': vis_phase_lock_left}
    
    
    region_df = pd.DataFrame(data = region_dict)
    clean_df = region_df[region_df["SleepStage"].isin([0,1,2])]
    print(clean_df)
    feature_df_train.append(clean_df)
    

feature_df_val = []
for animal in final_val_set:
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
    
    #somatosensory 
    soma_hfd = np.load(soma_hfd_dir + animal + '_hfd_concat.npy')
    soma_hfd_avg = [value[0] for value in soma_hfd]
    soma_hurst = np.load(soma_hurst_dir + animal + '_hurst_concat.npy')
    soma_hurst_avg = [value[0] for value in soma_hurst]
    
    soma_dispen = np.load(soma_dispen_dir + animal + '_dispen.npy')
    soma_gamma = np.load(soma_gamma_dir + animal + '_power.npy') 
    
    #somatosensory 
    vis_hfd = np.load(vis_hfd_dir + animal + '_hfd_concat.npy')
    vis_hfd_avg = [value[0] for value in vis_hfd]
    vis_hurst = np.load(vis_hurst_dir + animal + '_hurst_concat.npy')
    vis_hurst_avg = [value[0] for value in vis_hurst]
    
    vis_dispen = np.load(vis_dispen_dir + animal + '_dispen.npy')
    vis_gamma = np.load(vis_gamma_dir + animal + '_power.npy') 
    
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
    
    
    if animal in GRIN_het_IDs:
        genotype = 1
    elif animal in GRIN_wt_IDs:
        genotype = 0
        

    region_dict = {'Genotype': [genotype]*len(motor_dispen), 'SleepStage': br_state,
                   'Motor_DispEn': motor_dispen, 'Motor_Hurst': motor_hurst_avg, 
                   'Motor_HFD': motor_hfd_avg,'Motor_Gamma': motor_gamma, 
                   'Soma_DispEn': soma_dispen,'Soma_Hurst': soma_hurst_avg,
                   'Soma_HFD': soma_hfd_avg,'Soma_Gamma': soma_gamma,
                   'Visual_DispEn': vis_dispen,'Visual_Hurst': vis_hurst_avg,
                   'Visual_HFD': vis_hfd_avg, 'Vis_Gamma': vis_gamma, 
                   'Mot_CC_Right': mot_cross_corr_right, 'Mot_CC_Left': mot_cross_corr_left,
                   'Som_CC_Right': som_cross_corr_right, 'Som_CC_Left': som_cross_corr_left,
                   'Vis_CC_Right': vis_cross_corr_right, 'Mot_CC_Left': vis_cross_corr_left,
                   'Mot_PL_Right': mot_phase_lock_right, 'Mot_PL_Left': mot_phase_lock_left,
                   'Som_PL_Right': som_phase_lock_right, 'Som_PL_Left': som_phase_lock_left,
                   'Vis_PL_Right': vis_phase_lock_right, 'Vis_PL_Left': vis_phase_lock_left}
    
    
    region_df = pd.DataFrame(data = region_dict)
    clean_df = region_df[region_df["SleepStage"].isin([0,1,2])]
    print(clean_df)
    feature_df_val.append(clean_df)   
    
feature_df_test = []
for animal in final_test_set:
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
    
    #somatosensory 
    soma_hfd = np.load(soma_hfd_dir + animal + '_hfd_concat.npy')
    soma_hfd_avg = [value[0] for value in soma_hfd]
    soma_hurst = np.load(soma_hurst_dir + animal + '_hurst_concat.npy')
    soma_hurst_avg = [value[0] for value in soma_hurst]
    
    soma_dispen = np.load(soma_dispen_dir + animal + '_dispen.npy')
    soma_gamma = np.load(soma_gamma_dir + animal + '_power.npy') 
    
    #somatosensory 
    vis_hfd = np.load(vis_hfd_dir + animal + '_hfd_concat.npy')
    vis_hfd_avg = [value[0] for value in vis_hfd]
    vis_hurst = np.load(vis_hurst_dir + animal + '_hurst_concat.npy')
    vis_hurst_avg = [value[0] for value in vis_hurst]
    
    vis_dispen = np.load(vis_dispen_dir + animal + '_dispen.npy')
    vis_gamma = np.load(vis_gamma_dir + animal + '_power.npy') 
    
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
    

    
    if animal in GRIN_het_IDs:
        genotype = 1
    elif animal in GRIN_wt_IDs:
        genotype = 0
        

    region_dict = {'Genotype': [genotype]*len(motor_dispen), 'SleepStage': br_state,
                   'Motor_DispEn': motor_dispen, 'Motor_Hurst': motor_hurst_avg, 
                   'Motor_HFD': motor_hfd_avg,'Motor_Gamma': motor_gamma, 
                   'Soma_DispEn': soma_dispen,'Soma_Hurst': soma_hurst_avg,
                   'Soma_HFD': soma_hfd_avg,'Soma_Gamma': soma_gamma,
                   'Visual_DispEn': vis_dispen,'Visual_Hurst': vis_hurst_avg,
                   'Visual_HFD': vis_hfd_avg, 'Vis_Gamma': vis_gamma, 
                   'Mot_CC_Right': mot_cross_corr_right, 'Mot_CC_Left': mot_cross_corr_left,
                   'Som_CC_Right': som_cross_corr_right, 'Som_CC_Left': som_cross_corr_left,
                   'Vis_CC_Right': vis_cross_corr_right, 'Mot_CC_Left': vis_cross_corr_left,
                   'Mot_PL_Right': mot_phase_lock_right, 'Mot_PL_Left': mot_phase_lock_left,
                   'Som_PL_Right': som_phase_lock_right, 'Som_PL_Left': som_phase_lock_left,
                   'Vis_PL_Right': vis_phase_lock_right, 'Vis_PL_Left': vis_phase_lock_left}
    
    
    region_df = pd.DataFrame(data = region_dict)
    clean_df = region_df[region_df["SleepStage"].isin([0,1,2])]
    print(clean_df)
    feature_df_test.append(clean_df)
    
feature_concat_train = pd.concat(feature_df_train)
feature_concat_val = pd.concat(feature_df_val)
feature_concat_test = pd.concat(feature_df_test)

X_train = feature_concat_train.iloc[:, 1:]
y_train = feature_concat_train.iloc[:, 0]
X_val = feature_concat_val.iloc[:, 1:]
y_val = feature_concat_val.iloc[:, 0]
X_test = feature_concat_test.iloc[:, 1:]
y_test = feature_concat_test.iloc[:, 0]

undersample = RandomUnderSampler(sampling_strategy = 'majority')
X_train_new, y_train_new = undersample.fit_resample(X_train, y_train)
X_val_new, y_val_new = undersample.fit_resample(X_val, y_val)
X_test_new, y_test_new = undersample.fit_resample(X_test, y_test)


params = {'random_state': 42}

rounds = [{'max_depth': hp.quniform('max_depth', 1, 8, 1), #tree
            'min_child_weight': hp.loguniform('min_child_weight', -2, 3),
            'subsample': hp.uniform('subsample', 0.5, 1), #stochastic
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'reg_alpha': hp.uniform('reg_alpha', 0, 10), 
            'reg_lambda': hp.uniform('reg_lambda', 1, 10),
            'gamma': hp.loguniform('gamma', -10, 10),
            'learning_rate': hp.loguniform('learning_rate', -7, 0), 
            'n_estimators':hp.randint('n_estimators',200,1000)
           }]

all_trials = []
for round in rounds:
    params = {**params, **round}
    trials = Trials()
    best = fmin(fn=lambda space: hyperparameter_tuning(space, X_train_new,
                                                            y_train_new, X_val_new, y_val_new),
                space = params,
                algo = tpe.suggest,
                max_evals = 200,
                trials = trials,
               )
    params = {**params, **best}
    all_trials.append(trials)

os.chdir('/home/melissa/RESULTS/XGBoost')
all_trials = np.array(all_trials)
np.save('hyperparameter_tuning_parameters.npy', all_trials)