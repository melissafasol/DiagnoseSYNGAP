import os
import numpy as np 
import pandas as pd 
import xgboost as xgb

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

#Add genotypes
all_animals = pd.read_csv('/home/melissa/RESULTS/FINAL_MODEL/Rat/all_measures_xgboost.csv')
all_animals.drop(['Unnamed: 0'], axis = 1, inplace = True)

wt_ids = ['S7068', 'S7070', 'S7071', 'S7074', 'S7086', 'S7091', 'S7098', 'S7101'] #'S7087',
gap_ids = ['S7063', 'S7064', 'S7069', 'S7072', 'S7075', 'S7076', 'S7088', 'S7092', 'S7094',  'S7096']

def determine_genotype(animal_id, wt_ids, gap_ids):
    if animal_id in wt_ids:
        return 0  # WT
    elif animal_id in gap_ids:
        return 1  # GAP
    else:
        return None  # In case the ID is not found in either list

# Apply the function to each row in the DataFrame and ensure the type is integer
all_animals['Genotype'] = all_animals['Animal_ID'].apply(lambda x: determine_genotype(x, wt_ids, gap_ids)).astype('Int64')

# Move 'Genotype' column to the first position
cols = all_animals.columns.tolist()
cols.insert(0, cols.pop(cols.index('Genotype')))
all_animals = all_animals[cols]

accepted = ['Genotype', 'Animal_ID', 'Idx', 'Mot_CC_Gamma', 'Somatosensory_coh_gamma', 'Vis_CC_Beta', 
'Somatosensory_plv_theta', 'Visual_plv_beta', 'Visual_plv_sigma',
'Mot_CC_Beta', 'Som_CC_Gamma', 'Vis_Mot_pli_sigma', 'Vis_Mot_wpli_sigma',
'Visual_wpli_sigma', 'Vis_Mot_coh_beta', 'Somatosensory_coh_beta',
'Vis_Soma_plv_sigma', 'Somatosensory_wpli_beta', 'Vis_CC_Theta',
'Som_CC_Sigma', 'Soma_Motor_pli_beta', 'Som_CC_Beta', 'Visual_coh_beta',
'Visual_coh_gamma', 'Mot_Disp', 'Soma_Motor_pli_gamma', 'Somatosensory_wpli_gamma',
'Vis_Mot_plv_gamma', 'Somatosensory_pli_beta', 'Soma_Motor_coh_gamma', 'Motor_wpli_gamma',
'Somatosensory_plv_gamma', 'Soma_Motor_plv_beta', 'Somatosensory_plv_delta',
'Visual_plv_gamma', 'Mot_CC_Delta', 'Som_CC_Theta', 'Somatosensory_coh_theta',
'Vis_Mot_pli_gamma', 'Vis_Soma_pli_gamma', 'Vis_Mot_plv_sigma', 'Vis_Soma_plv_theta',
'Motor_wpli_theta', 'Vis_Soma_coh_delta', 'Vis_Soma_plv_gamma', 'Motor_pli_beta',
'Visual_pli_sigma', 'Vis_Soma_plv_delta', 'Motor_pli_theta', 'Vis_CC_Gamma',
'Soma_Motor_plv_gamma', 'Vis_Mot_coh_theta', 'Vis_Soma_coh_theta', 'Motor_coh_gamma',
'Som_Disp', 'Somatosensory_coh_sigma', 'Mot_HFD', 'Vis_Soma_coh_beta', 'Visual_coh_sigma',
'Somatosensory_plv_sigma', 'Mot_CC_Theta', 'Somatosensory_pli_gamma', 'Vis_Soma_plv_beta',
'Visual_pli_gamma', 'Vis_CC_Sigma', 'Vis_Mot_coh_gamma', 'Vis_Disp', 'Soma_Motor_coh_beta',
'Som_Hurst', 'Vis_Hurst', 'Vis_Mot_pli_beta', 'Vis_Mot_pli_theta', 'Som_HFD', 'Motor_pli_sigma',
'Vis_Soma_coh_gamma', 'Vis_Soma_coh_sigma', 'Vis_CC_Delta', 'Vis_Mot_plv_theta', 'Som_CC_Delta',
'Motor_plv_gamma', 'Vis_HFD', 'Somatosensory_plv_beta', 'Mot_Hurst', 'Motor_pli_gamma',
'Vis_Mot_plv_beta']

selected_columns = all_animals[accepted]

# Combine the two lists and create a list of labels (0 for human_wt and 1 for human_gap)
all_ids = np.unique(selected_columns['Animal_ID'].to_list())
labels = [0] * len(wt_ids) + [1] * len(gap_ids)

# Split the combined list into training and test sets, stratifying by the labels
train_ids, test_ids,_, _ = train_test_split(all_ids, labels, test_size=0.3, stratify=labels, random_state=10)

X_train = selected_columns[selected_columns["Animal_ID"].isin(train_ids)]
X_test = selected_columns[selected_columns["Animal_ID"].isin(test_ids)]


group_by_patient_id = X_train.groupby(['Animal_ID'])
groups_by_patient_id_list = np.array(X_train['Animal_ID'].values)

X_train_new = X_train.iloc[:, 3:]
X_test_new = X_test.iloc[:, 3:]
y_train = X_train.iloc[:, 0]
y_test = X_test.iloc[:, 0]

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
    
    space['max_depth'] = int(space['max_depth'])
    
    # Store the cross-validated AUC-ROC scores
    cv_scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = xgb.XGBClassifier( **space)
        model.fit(X_train, y_train)

        # Predict probabilities for the positive class (usually column index 1)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Compute AUC-ROC score and append to cv_scores
        score = roc_auc_score(y_test, y_pred_proba)
        cv_scores.append(score)

    # Use the average AUC-ROC score across all folds
    average_cv_score = np.mean(cv_scores)
    
    return {'loss': -average_cv_score, 'status': STATUS_OK}

# Hyperopt settings
trials = Trials()
best = fmin(fn=lambda space: hyperparameter_tuning(space, X_train_new, y_train, n_splits=3),
            space=options,
            algo=tpe.suggest,
            max_evals=250,
            trials=trials)