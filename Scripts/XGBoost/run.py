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


train_2 = prepare_df_2(train_2_ids, br_directory, motor_hfd_dir, motor_hurst_dir, motor_dispen_dir, 
                 motor_gamma_dir, motor_theta_dir, soma_hfd_dir, soma_hurst_dir, soma_dispen_dir,
                 soma_gamma_dir, soma_theta_dir, vis_hfd_dir, vis_hurst_dir, vis_dispen_dir,
                vis_gamma_dir, vis_theta_dir, SYNGAP_het, SYNGAP_wt)

train_1 = prepare_df_1(train_1_ids, br_directory, motor_hfd_dir, motor_hurst_dir, motor_dispen_dir, 
                 motor_gamma_dir, motor_theta_dir, soma_hfd_dir, soma_hurst_dir, soma_dispen_dir,
                 soma_gamma_dir, soma_theta_dir, vis_hfd_dir, vis_hurst_dir, vis_dispen_dir,
                vis_gamma_dir, vis_theta_dir, SYNGAP_het, SYNGAP_wt)

val_2 = prepare_df_2(val_2_ids, br_directory, motor_hfd_dir, motor_hurst_dir, motor_dispen_dir, 
                 motor_gamma_dir, motor_theta_dir, soma_hfd_dir, soma_hurst_dir, soma_dispen_dir,
                 soma_gamma_dir, soma_theta_dir, vis_hfd_dir, vis_hurst_dir, vis_dispen_dir,
                vis_gamma_dir, vis_theta_dir, SYNGAP_het, SYNGAP_wt)

val_1 = prepare_df_1(val_1_ids, br_directory, motor_hfd_dir, motor_hurst_dir, motor_dispen_dir, 
                 motor_gamma_dir, motor_theta_dir, soma_hfd_dir, soma_hurst_dir, soma_dispen_dir,
                 soma_gamma_dir, soma_theta_dir, vis_hfd_dir, vis_hurst_dir, vis_dispen_dir,
                vis_gamma_dir, vis_theta_dir, SYNGAP_het, SYNGAP_wt)

test_2 = prepare_df_2(test_2_ids, br_directory, motor_hfd_dir, motor_hurst_dir, motor_dispen_dir, 
                 motor_gamma_dir, motor_theta_dir, soma_hfd_dir, soma_hurst_dir, soma_dispen_dir,
                 soma_gamma_dir, soma_theta_dir, vis_hfd_dir, vis_hurst_dir, vis_dispen_dir,
                vis_gamma_dir, vis_theta_dir, SYNGAP_het, SYNGAP_wt)

test_1 = prepare_df_1(test_1_ids, br_directory, motor_hfd_dir, motor_hurst_dir, motor_dispen_dir, 
                 motor_gamma_dir, motor_theta_dir, soma_hfd_dir, soma_hurst_dir, soma_dispen_dir,
                 soma_gamma_dir, soma_theta_dir, vis_hfd_dir, vis_hurst_dir, vis_dispen_dir,
                vis_gamma_dir, vis_theta_dir, SYNGAP_het, SYNGAP_wt)


feature_concat_train = pd.concat([train_1, train_2])
feature_concat_val = pd.concat([val_1, val_2])
feature_concat_test = pd.concat([test_1, test_2])

X_train = feature_concat_train.iloc[:, 1:]
y_train = feature_concat_train.iloc[:, 0]
X_val = feature_concat_val.iloc[:, 1:]
y_val = feature_concat_val.iloc[:, 0]
X_test = feature_concat_test.iloc[:, 1:]
y_test = feature_concat_test.iloc[:, 0]

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