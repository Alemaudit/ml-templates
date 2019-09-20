# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:43:47 2019

@author: aless
"""

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def tune_hyperparameters(X, y, estimator, param_distributions, n_iter=40,
                         scoring=roc_auc_score, n_folds=5):

    cv_ = StratifiedKFold(n_splits=n_folds, shuffle=True)

    tuner = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        n_jobs=-1,
        iid=False,
        cv=cv_)

    tuner.fit(X, y)

    return tuner
