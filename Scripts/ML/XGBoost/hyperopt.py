import xgboost as xgb
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Union, Any, Callable

def hyperparameter_tuning(space: Dict[str, Union[float, int]],
                         X_train: pd.DataFrame, y_train: pd.Series, 
                         X_test: pd.DataFrame, y_test: pd.Series, 
                         early_stopping_rounds: int = 50, 
                         metric: Callable = accuracy_score) -> Dict[str, Any]:
    
    '''Perform hyperparameter tuning for an XGBoost classifier. 
    
    This function takes a dictionary of hyperparameters, training and test data, and an optional value
    for early stopping rounds, and returns a dictionary with the loss and model resulting from 
    the tuning process. The model is trained using the training data and evaluated on the test 
    data. The loss is computed as the negative of the accuracy score.
    
    space: Dict[str, Union[float, int]]
    A dictionary of hyperparameters for the XGBoost classifier
    
    X_train: pd.DataFrame
    The training data
    
    y_train: pd.Series
    The training target
    
    X_test: pd.Dataframe
    The test data
    
    y_test: pd.Series
    The test target
    
    early_stopping rounds: int, optional 
    The number of early stopping rounds to use. The default is 50
    
    metric: callable
    Metric to maximize. Default is accuracy
    
    Returns: 
    Dict[str, Any]
        A dictionary with the loss and model resulting from the tuning process. 
        The loss is a float, and the model is an XGBoost classifier'''
    
    int_vals = ['max_depth', 'n_estimators']
    
    space = {k: (int(val) if k in int_vals else val)
            for k, val in space.items()}
    
    model = xgb.XGBClassifier(**space)
    evaluation = [(X_train, y_train), 
                  (X_test, y_test)]
    
    model.fit(X_train, y_train, eval_set=evaluation, 
              early_stopping_rounds=early_stopping_rounds, 
              verbose=False)
    
    y_pred = model.predict(X_test)
    score = metric(y_test, y_pred)
    
    return {'loss': -score, 'status': STATUS_OK, 'model': model}
