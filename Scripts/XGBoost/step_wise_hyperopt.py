import pandas as pd

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials 
from sklearn.metrics import accuracy_score, roc_auc_score

from typing import Any, Dict, Union
import xgboost as xgb


def hyperparameter_tuning(space:Dict[str, Union[float, int]],
                          X_train: pd.DataFrame, y_train:pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          early_stopping_rounds: int=50,
                          metric: callable = accuracy_score) -> Dict[str, Any]:
    
    """Perform hyperparameter tuning for XGBoost classifier.
    
    Input = dictionary of hyperparameters, training and test data, 
    and an optional value for early stopping rounds, and returns a 
    dictionary with the loss and model resulting from the tuning
    process. The model is trained using the training data and 
    evaluated on the test data. The loss is computed as the negative 
    of the accuracy score.
    
    Parameters
    ----------
    space : Dict[str, Union[float, int]] - a dictionary of hyperparameters for the XGBoost classifier
    X_train: pd.DataFrame (training data)
    y_train: pd.Series (training target)
    
    X_test: pd.DataFrame (test data)
    y_test: pd.Series (test target)
    
    early_stopping_rounds: int, optional 
    The number of early stopping rounds to use. The default value is 50 
    metric: callable 
    Metric to maximise. Default is accuracy 
    
    Returns
    -------
    Dict[str, Any]
        A dictionary with the loss and model resulting from the tuning process. The loss is a float, and the
        model is a XGBoost classifier. 
    """
    int_vals = ['max_depth', 'reg_alpha ']
    space = {k: (int(val) if k in int_vals else val)
             for k, val in space.items()}
    space['early_stopping_rounds'] = early_stopping_rounds
    model = xgb.XGBClassifier(**space)
    evaluation = [(X_train, y_train),
                  (X_test, y_test)]
    model.fit(X_train, y_train, eval_set = evaluation, verbose = False)
    pred = model.predict(X_test)
    score = metric(y_test, pred)
    return {'loss': -score, 'status': STATUS_OK, 'model': model}
    

params = {'random_state': 42}

    
rounds = [{'max_depth': hp.quniform('max_depth', 1, 8, 1), 
            'min_child_weight': hp.loguniform('min_child_weight', -2, 3)},
            {'subsample': hp.uniform('subsample', 0.5, 1), 
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1)},
            {'reg_alpha': hp.uniform('reg_alpha', 0, 10), 
            'reg_lambda': hp.uniform('reg_lambda', 1, 10)},
            {'gamma': hp.loguniform('gamma', -10, 10)},
            {'learning_rate': hp.loguniform('learning_rate', -7, 0)}
            ]

