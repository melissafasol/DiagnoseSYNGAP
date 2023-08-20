'''Script to implement grouped k-fold cross validation to keep animal ids in train and val separate '''

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, GroupKFold


def group_k_fold(n_splits, x_train, y_train, groups_id):

    n_splits = 3
    group_kfold = GroupKFold(n_splits = n_splits)
    print(group_kfold.get_n_splits(x_train, y_train, groups = groups_id))


    result = []   
    for train_idx, val_idx in group_kfold.split(x_train, y_train, groups = groups_id):
        train_fold = x_train.iloc[train_idx]
        val_fold = x_train.iloc[val_idx]
        result.append((train_fold, val_fold))
    
    
    
    train_fold_1, val_fold_1 = result[0][0],result[0][1]
    train_fold_2, val_fold_2 = result[1][0],result[1][1]
    train_fold_3, val_fold_3 = result[2][0],result[2][1]

    return train_fold_1, train_fold_2, train_fold_3, val_fold_1, val_fold_2, val_fold_3