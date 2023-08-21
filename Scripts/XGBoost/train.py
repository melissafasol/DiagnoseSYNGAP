import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import hp, tpe, fmin, Trials
from copy import copy
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, \
    mean_squared_error, average_precision_score


class XGBtrain():
    
    def __init__(self, x_train, y_train, x_test, y_test, hyperparam_text_path):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test 
        self.hyperparam_text_path = hyperparam_text_path
        self.best_params = {
            'verbosity': 0,
            'eval_metric': 'aucpr',
            'max_bin': 64,
            'max_delta_step': 1,
            'early_stopping_rounds': 15,
            'objective': 'binary:logistic',
        }
        
        
    def train(self):
        '''
        Train XGBoost model with group k fold and hyperopt
        '''
        
        def _objective(args):
            model = xgb.train(args, xy, int(args['n_estimators']))
            x_test_m = xgb.DMatrix(self.x_test)
            y_pred = model.predict(x_test_m)
            
            output = average_precision_score(self.y_test, y_pred)
            
            return output
        
        xy = xgb.DMatrix(self.x_train, label= self.y_train)
        
        learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
        n_estimators = [100, 300, 500, 1000]
        max_depth = np.logspace(0.5, 2, 10, dtype=int)
        min_child_weight = np.arange(0, 5, 1, dtype=int)
        params = copy(self.best_params)
        params.update({
            'learning_rate': hp.choice('learning_rate', learning_rate),
            'n_estimators': hp.choice('n_estimators', n_estimators),
            'max_depth': hp.choice('max_depth', max_depth),
            'min_child_weight': hp.choice('min_child_weight',
                                          min_child_weight),
            'gamma': hp.quniform('gamma', 0, 20, 1),
            'lambda': hp.quniform('lambda', 0, 10, 0.1),
            'alpha': hp.quniform('alpha', 0, 60, 1),
            'subsample': hp.quniform('subsample', 0.3, 1, 0.05),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.3, 0.7, 0.05)
        })
        

        trials = Trials()
        best = fmin(fn=_objective,
                    space=params,
                    algo=tpe.suggest,
                    max_evals=20,
                    trials=trials)

        print(best)
        with open(str(self.hyperparam_text_path) + '/best_params.txt', 'w') as f:
            f.write(str(best))
            self.best_params.update({
            'learning_rate': learning_rate[best['learning_rate']],
            'n_estimators': n_estimators[best['n_estimators']],
            'max_depth': max_depth[best['max_depth']],
            'min_child_weight': min_child_weight[best['min_child_weight']],
            'gamma': best['gamma'],
            'lambda': best['lambda'],
            'alpha': best['alpha'],
            'subsample': best['subsample'],
            'colsample_by==tree': best['colsample_bytree']
        })
        print(self.best_params)
        model = xgb.train(self.best_params, xy)
        print(model)
        
        
        def predict(self, x_data, y_data) -> pd.Series:
            
            """
            Predict output for provided data.

            Parameters
            ----------
            data: pandas.DataFrame
                input data
            return_proba: bool
                a flag: if true returns probability of each class instead of
                prediction

            Returns
            -------
            pandas.Series
            series with predictions or probabilities of prediction of each
            class
            """
            
            features_names = x_data.columns.values
            y_pred = self.model.predict(xgb.DMatrix(data = x_data, 
                                                    feature_names = features_names))
            
            tmp = np.round(y_pred)
            mse = mean_squared_error(y_data, tmp)
            accuracy = accuracy_score(y_data, tmp)
            precision = precision_score(y_data, tmp, average="weighted")
            try:
                roc = roc_auc_score(y_data, tmp)
                prauc = average_precision_score(y_data, y_pred)
            except ValueError:
                prauc = roc = np.nan
                
            return y_pred, {
                'mse': mse,
                'roc': roc,
                'accuracy': accuracy,
                'precision': precision,
                'prauc': prauc
                 }
