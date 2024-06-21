import os
import numpy as np 
import pandas as pd 
import xgboost as xgb
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

import xgbfir
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import dtreeviz
from typing import Any, Dict, Union

from yellowbrick import model_selection as ms
from yellowbrick.model_selection import validation_curve

from sklearn import metrics


patient_list  =  ['P1 N1', 'P2 N1', 'P2 N2', 'P3 N1', 'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1',
                   'P6 N2', 'P7 N1', 'P7 N2','P8 N1','P10 N1', 'P11 N1', 'P15 N1',
                  'P16 N1', 'P17 N1', 'P18 N1','P20 N1', 'P21 N1', 'P21 N2', 'P21 N3',
                  'P22 N1','P23 N1', 'P23 N3', 'P24 N1','P27 N1','P28 N1',
                  'P28 N2', 'P29 N2', 'P30 N1'] 

human_wt = ['P1', 'P11', 'P17', 'P18', 'P21', 'P24', 'P27','P28', 'P29', 'P4']
human_gap = ['P3','P10', 'P15', 'P16', 'P2', 'P5', 'P6', 'P7','P8', 'P20',  'P22',
            'P23', 'P30'] 

#load all dataframes 
conn_mne = '/home/melissa/RESULTS/FINAL_MODEL/Human/Connectivity_MNE/xgb_dataframes/'
cc_mne = '/home/melissa/RESULTS/FINAL_MODEL/Human/Cross_Corr_Channels/'
hfd_mne = '/home/melissa/RESULTS/FINAL_MODEL/Human/Complexity/hfd_df/'
hurst_mne = '/home/melissa/RESULTS/FINAL_MODEL/Human/Complexity/hurst_df/'
disp_mne = '/home/melissa/RESULTS/FINAL_MODEL/Human/Complexity/DispEn_DF/'

delta_plv_graph_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Graph_Theory/plv/delta/'
theta_plv_graph_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Graph_Theory/plv/theta/'
sigma_plv_graph_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Graph_Theory/plv/sigma/'
beta_plv_graph_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Graph_Theory/plv/beta/'

delta_pli_graph_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Graph_Theory/pli/delta/'
theta_pli_graph_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Graph_Theory/pli/theta/'
sigma_pli_graph_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Graph_Theory/pli/sigma/'
beta_pli_graph_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Graph_Theory/pli/beta/'

delta_wpli_graph_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Graph_Theory/wpli/delta/'
theta_wpli_graph_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Graph_Theory/wpli/theta/'
sigma_wpli_graph_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Graph_Theory/wpli/sigma/'
beta_wpli_graph_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Graph_Theory/wpli/beta/'

delta_coh_graph_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Graph_Theory/coh/delta/'
theta_coh_graph_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Graph_Theory/coh/theta/'
sigma_coh_graph_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Graph_Theory/coh/sigma/'
beta_coh_graph_dir = '/home/melissa/RESULTS/FINAL_MODEL/Human/Graph_Theory/coh/beta/' 

all_dataframes = []

for patient in patient_list:
    conn_df = pd.read_csv(conn_mne + str(patient) + '_all_conn_measures.csv')
    cc_df = pd.read_csv(cc_mne + str(patient) + '.csv')
    hfd_df = pd.read_csv(hfd_mne + str(patient) + '_hfd.csv')
    hurst_df = pd.read_csv(hurst_mne + str(patient) + '_hurst.csv')
    disp_df = pd.read_csv(disp_mne + str(patient) + '_dispen.csv')
    
    delta_plv = pd.read_csv(delta_plv_graph_dir + str(patient) + '_delta_graph_theory.csv' )
    theta_plv = pd.read_csv(theta_plv_graph_dir + str(patient) + '_theta_graph_theory.csv' )
    sigma_plv = pd.read_csv(sigma_plv_graph_dir + str(patient) + '_sigma_graph_theory.csv' )
    beta_plv  = pd.read_csv(beta_plv_graph_dir + str(patient) + '_beta_graph_theory.csv' )
    
    delta_pli = pd.read_csv(delta_pli_graph_dir + str(patient) + '_delta_graph_theory.csv' )
    theta_pli = pd.read_csv(theta_pli_graph_dir + str(patient) + '_theta_graph_theory.csv' )
    sigma_pli = pd.read_csv(sigma_pli_graph_dir + str(patient) + '_sigma_graph_theory.csv' )
    beta_pli  = pd.read_csv(beta_pli_graph_dir + str(patient) + '_beta_graph_theory.csv' )
    
    delta_wpli = pd.read_csv(delta_wpli_graph_dir + str(patient) + '_delta_graph_theory.csv' )
    theta_wpli = pd.read_csv(theta_wpli_graph_dir + str(patient) + '_theta_graph_theory.csv' )
    sigma_wpli = pd.read_csv(sigma_wpli_graph_dir + str(patient) + '_sigma_graph_theory.csv' )
    beta_wpli  = pd.read_csv(beta_wpli_graph_dir + str(patient) + '_beta_graph_theory.csv' )
    
    delta_coh = pd.read_csv(delta_coh_graph_dir + str(patient) + '_delta_graph_theory.csv' )
    theta_coh = pd.read_csv(theta_coh_graph_dir + str(patient) + '_theta_graph_theory.csv' )
    sigma_coh = pd.read_csv(sigma_coh_graph_dir + str(patient) + '_sigma_graph_theory.csv' )
    beta_coh  = pd.read_csv(beta_coh_graph_dir + str(patient) + '_beta_graph_theory.csv' )
    
    # Store them in a list
    dfs = [conn_df, cc_df, hfd_df, hurst_df, disp_df, delta_plv, theta_plv, sigma_plv, beta_plv,
          delta_pli, theta_pli, sigma_pli, beta_pli, delta_wpli, theta_wpli, sigma_wpli, beta_wpli,
          delta_coh, theta_coh, sigma_coh, beta_coh]

    # Check for 'Unnamed: 0' column and drop it if it exists
    dfs = [df.drop('Unnamed: 0', axis=1) if 'Unnamed: 0' in df.columns else df for df in dfs]
    
    # Extract them back
    #conn, cc, hfd, hurst, disp, delta_plv, theta_plv, sigma = dfs
    all_measures = pd.concat(dfs, axis = 1)
    all_dataframes.append(all_measures)
    
concat_all_dtaframes = pd.concat(all_dataframes, axis = 0)
#remove any NaN values
df_cleaned = concat_all_dtaframes.dropna()


#group IDs
df_cleaned['Patient_ID'] = df_cleaned['Patient_ID'].str.split().str[0]
# Combine the two lists and create a list of labels (0 for human_wt and 1 for human_gap)
all_ids = np.unique(df_cleaned['Patient_ID'].to_list())
labels = [0] * len(human_wt) + [1] * len(human_gap)

# Split the combined list into training and test sets, stratifying by the labels
train_ids, test_ids,_, _ = train_test_split(all_ids, labels, test_size=0.3, stratify=labels, random_state= 2)

#ensure Genotype is 0 and 1
df_cleaned['Genotype'] = df_cleaned['Genotype'].map({'WT': 0, 'GAP': 1})

#Results from BorutaSHAP
accepted_columns = ['Idx', 'Genotype', 'Patient_ID',
                    'cc_E2_M2', 'coh_theta_O1_E2', 'C3_RDE',
                    'plv_delta_O1_C3', 'cc_F3_C3', 'pli_sigma_O1_C3',
                    'O1_dispen', 'plv_beta_O1_E1', 'cc_E2_F3',
                    'coh_sigma_O1_E1', 'coh_theta_O1_F3', 
                    'plv_delta_C3_F3', 'plv_theta_O1_E1', 'cc_E2_C3',
                    'wpli_beta_C3_F3', 'hfd_chan_F3', 'modularity_beta_coh',
                    'hfd_E2', 'cc_O1_M2', 'cc_E2_O1', 'hfd_E1', 'cc_F3_M2',
                    'coh_beta_E2_E1', 'cc_E1_E2', 'wpli_sigma_E2_E1', 'hfd_chan_M2',
                    'modularity_beta_plv', 'C3_dispen', 'cc_E1_C3', 'plv_theta_O1_F3',
                    'coh_theta_O1_E1', 'cc_C3_O1', 'plv_theta_E2_E1', 'cc_E1_F3', 
                    'cc_E1_O1', 'cc_F3_O1', 'plv_sigma_E2_E1', 'coh_delta_C3_F3']

selected_columns = df_cleaned[accepted_columns]

#train and test sets
X_train = selected_columns[selected_columns["Patient_ID"].isin(train_ids)]
X_test = selected_columns[selected_columns["Patient_ID"].isin(test_ids)]

group_by_patient_id = X_train.groupby(['Patient_ID'])
groups_by_patient_id_list = np.array(X_train['Patient_ID'].values)

#final data sets for training 
X_train_new = X_train.iloc[:, 3:]
X_test_new = X_test.iloc[:, 3:]
y_train = X_train.iloc[:, 1]
y_test = X_test.iloc[:, 1]

#parameters to tune 
options = {
    'max_depth': hp.quniform('max_depth', 1, 8, 1),
    'min_child_weight': hp.loguniform('min_child_weight', -2, 3),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0, 10),
    'reg_lambda': hp.uniform('reg_lambda', 1, 10),
    'gamma': hp.loguniform('gamma', -10, 10),
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'n_estimators': hp.choice('n_estimators', range(50, 1001, 50)),
    'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 100),
    'max_delta_step': hp.quniform('max_delta_step', 0, 10, 1),
    'tree_method': 'exact', 
    #'sample_type': hp.choice('sample_type', ['uniform', 'weighted']),
    #'normalize_type': hp.choice('normalize_type', ['tree', 'forest']),
    #'rate_drop': hp.uniform('rate_drop', 0, 1),
    #'skip_drop': hp.uniform('skip_drop', 0, 1),
    'random_state': 9
}

def hyperparameter_tuning(space, X, y, n_splits=3):
    # Initialize cross-validation scheme
    cv = StratifiedKFold(n_splits=n_splits)
    
    # Convert max_depth to integer if it's a float
    space['max_depth'] = int(space['max_depth'])
    
    # Store the cross-validated AUC-ROC scores along with fold number
    cv_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = xgb.XGBClassifier( **space)
        model.fit(X_train, y_train)

        # Predict probabilities for the positive class (usually column index 1)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Compute AUC-ROC score
        score = roc_auc_score(y_test, y_pred_proba)
        
        # Append the fold number and score to cv_scores
        cv_scores.append((fold, score))

    # Use the average AUC-ROC score across all folds
    average_cv_score = np.mean([score for _, score in cv_scores])
    
    # Return the average score, individual fold scores, and fold numbers
    return {'loss': average_cv_score, 'status': STATUS_OK, 'fold_scores': cv_scores}


# Hyperopt settings
trials = Trials()
best = fmin(fn=lambda space: hyperparameter_tuning(space, X_train_new, y_train, n_splits=3),
            space=options,
            algo=tpe.suggest,
            max_evals=250,
            trials=trials)

# Save the Trials object to a file
with open('/home/melissa/RESULTS/FINAL_MODEL/' + 'hyperopt_trials_accepted_boruta_human_dart.pkl', 'wb') as f:
    pickle.dump(trials, f)

