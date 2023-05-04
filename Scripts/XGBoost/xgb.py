import numpy as np
import pandas as pd
import xgboost as xgb
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from copy import copy
from typing import List
from hyperopt import hp, tpe, fmin, Trials
from sklearn.preprocessing import LabelEncoder
from pyseizure.helpers.iterator import Iterator
from pyseizure.classifier.classifier import Classifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, \
    mean_squared_error, average_precision_score

from pyseizure.preprocessing.feature_selection.feature_selector import \
    FeatureSelector

VERBOSITY = 0
TREE_METHOD = 'gpu_hist'


class XGB(Classifier):
    def __init__(self, train_samples: List[str], test_samples: List[str],
                 labels: List['str'] = None, binary: bool = False,
                 feature_selector: FeatureSelector = None):
        super().__init__(train_samples, test_samples, feature_selector)
        self.model = None
        self.binary = binary
        self.label_encoder = LabelEncoder()
        if labels is not None:
            self.label_encoder.fit(labels)
        self.best_params = {
            'verbosity': VERBOSITY,
            'gpu_id': 3,
            'eval_metric': 'aucpr',
            'tree_method': TREE_METHOD,
            'max_bin': 64,
            'max_delta_step': 1,
            'early_stopping_rounds': 15,
            'objective': 'binary:logistic' if binary else 'multi:softmax',
            'num_class': None if binary else self.label_encoder.classes_.size,
        }

    def train(self):
        """
        Train XGBoost model.

        This function tunes hyperparameters, saves the best in the object
        variable, and trains model using them.

        Returns
        -------
        None
        """
        def _objective(args):
            model = xgb.train(args, xy, int(args['n_estimators']))
            y_pred = model.predict(x_test, output_margin=not self.binary)
            if self.binary:
                output = average_precision_score(y_test, y_pred)
            else:
                output = average_precision_score(y_test, y_pred[:, 1])

            return output

        it = Iterator(self.train_samples, self.label_encoder,
                      self.feature_selector)
        xy = xgb.DeviceQuantileDMatrix(it, missing=np.NaN,
                                       enable_categorical=False)

        test = pq.read_table(self.test_samples[0]).to_pandas()
        x_test = test.iloc[:, :-1][it.features_names]
        y_test = test.iloc[:, -1]
        features_names = x_test.columns.values
        if self.feature_selector:
            features_names = it.features_names
        x_test = np.array(x_test, dtype='f')
        x_test = xgb.DeviceQuantileDMatrix(x_test,
                                           feature_names=features_names)
        y_test = self.label_encoder.transform(y_test)

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
        with open('output/best_params.txt', 'w') as f:
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
            'colsample_bytree': best['colsample_bytree']
        })
        print(self.best_params)
        self.model = xgb.train(self.best_params, xy)
        print(self.model)

    def predict(self, data: pd.DataFrame,
                return_proba: bool = False) -> pd.Series:
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
        y_test = self.label_encoder.transform(data.iloc[:, -1])
        x_test = data.loc[:, self.model.feature_names]
        features_names = x_test.columns.values
        x_test = np.array(x_test, dtype='f')
        y_pred = self.model.predict(
            xgb.DeviceQuantileDMatrix(data=x_test,
                                      feature_names=features_names),
            output_margin=False if self.binary else return_proba)

        if return_proba and not self.binary:
            tmp = np.where(y_pred[:, 0] > y_pred[:, 1], 0, 1)
        elif self.binary:
            tmp = np.round(y_pred)
        else:
            tmp = y_pred
        mse = mean_squared_error(y_test, tmp)
        accuracy = accuracy_score(y_test, tmp)
        precision = precision_score(y_test, tmp, average="weighted")
        try:
            roc = roc_auc_score(y_test, tmp)
            if self.binary:
                prauc = average_precision_score(y_test, y_pred)
            else:
                prauc = average_precision_score(y_test, y_pred[:, 1])
        except ValueError:
            prauc = roc = np.nan

        return y_pred, {
            'mse': mse,
            'roc': roc,
            'accuracy': accuracy,
            'precision': precision,
            'prauc': prauc
        }

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate model on test data.

        Function creates two figures:
            - most important features
            - the best iteration tree

        Returns
        -------
        pandas.DataFrame
            dataframe with each evaluation round
        """
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        xgb.plot_tree(self.model, num_trees=self.model.best_iteration,
                      fontsize=15, ax=ax)
        plt.savefig('output/xgb_tree.png')

        fig, ax = plt.subplots(figsize=(25, 25), dpi=300)
        xgb.plot_importance(self.model, ax=ax)
        plt.savefig('output/xgb_features.png')

        scores = []

        for sample in self.test_samples:
            x_test = pq.read_table(sample).to_pandas()
            _, score = self.predict(x_test, True)
            scores.append(score)
        result = pd.DataFrame(scores)
        result.to_csv('output/eval.csv')
        print(result.to_string())
        print('---mean---')
        print(result.mean(axis=0))

        return result