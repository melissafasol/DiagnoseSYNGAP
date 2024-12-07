import os
import numpy as np 
import pandas as pd 

from sklearn import model_selection, preprocessing 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.utils import resample
from sklearn.model_selection import GroupKFold
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from collections import Counter
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt

# Helper function to ensure both classes are present in a fold
def is_fold_valid(y_train, y_test):
    """Check if both classes are present in the test set."""
    return len(np.unique(y_train)) == 2 and len(np.unique(y_test)) == 2

# Load data
print("Loading datasets...")
wt_csv = pd.read_csv('/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Results/WT_all_features.csv', index_col=0)
gap_csv = pd.read_csv('/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Results/GAP_all_features.csv', index_col=0)
all_features = pd.concat([wt_csv, gap_csv], axis=0)
print("Datasets loaded and concatenated.")

important_features = pd.read_csv('/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Results/important_features_boruta_shap.csv')
important_features_list = important_features.iloc[:, 1].tolist()
print(f"Important features extracted: {important_features_list}")

X = all_features[important_features_list]
y = all_features['Genotype']
animal_ids = all_features['Animal_ID']
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

# Parameters for cross-validation and bootstrapping
n_folds = 5
n_bootstraps = 10

# Hyperopt space
space = {
    'max_depth': hp.choice('max_depth', np.arange(3, 10, dtype=int)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'n_estimators': hp.choice('n_estimators', np.arange(50, 500, 10, dtype=int)),
    'gamma': hp.uniform('gamma', 0, 1),
    'min_child_weight': hp.uniform('min_child_weight', 1, 10),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 10)
}

# Store metrics for each fold and bootstrap
results = []

# Objective function for Hyperopt
def objective(params):
    print("Running Hyperopt with parameters:", params)
    auc_scores = []
    group_kfold = GroupKFold(n_splits=n_folds)
    
    for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups=animal_ids)):
        print(f"Evaluating fold {fold_idx + 1}...")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        if not is_fold_valid(y_train, y_test):
            print(f"Invalid fold detected, skipping fold {fold_idx + 1}.")
            continue
        
        X_train, y_train = resample(X_train, y_train, random_state=42)
        ros = RandomOverSampler(random_state=42)
        X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
        
        model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train_res, y_train_res)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        auc_scores.append(auc)
        print(f"Fold {fold_idx + 1} AUC: {auc}")
    
    mean_auc = np.mean(auc_scores) if auc_scores else 0
    print(f"Mean AUC for parameters {params}: {mean_auc}")
    return {'loss': -mean_auc, 'status': STATUS_OK}

# Step 1: Hyperparameter tuning
print("Starting Hyperparameter Optimization...")
trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
print(f"Best parameters found: {best_params}")

# Step 2: Collect metrics using the best hyperparameters
group_kfold = GroupKFold(n_splits=n_folds)
print("Starting evaluation with bootstrapping...")
for b in range(n_bootstraps):
    print(f"Bootstrap iteration {b + 1}...")
    for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups=animal_ids)):
        print(f"Processing fold {fold_idx + 1}...")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        if not is_fold_valid(y_train, y_test):
            print(f"Invalid fold detected, skipping fold {fold_idx + 1}.")
            continue
        
        X_train, y_train = resample(X_train, y_train, random_state=b)
        ros = RandomOverSampler(random_state=42)
        X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
        
        final_model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
        final_model.fit(X_train_res, y_train_res)
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]
        y_pred = final_model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        results.append({
            'bootstrap_iteration': b + 1,
            'fold': fold_idx + 1,
            'roc_auc': auc,
            'confusion_matrix': cm,
            'accuracy': accuracy_score(y_test, y_pred)
        })
        print(f"Fold {fold_idx + 1} AUC: {auc}, Accuracy: {accuracy_score(y_test, y_pred)}")

# Save results
results_df = pd.DataFrame(results)
results_file = "/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Results/results_metrics.csv"
results_df.to_csv(results_file, index=False)
print(f"Metrics saved to {results_file}")
