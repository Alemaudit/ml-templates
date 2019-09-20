# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:56:31 2019

@author: aless
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

hyperparameters_spaces = {
    'LogisticRegression': {
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-3, 3, num=7)
    },
    'MLPClassifier': {
        'hidden_layer_sizes': [
            (10,), (50,), (100,), (50, 10,), (50, 50,), (100, 10,), (100, 50,),
            (100, 100,), (40, 40, 40, 20),
        ],
        'alpha': np.logspace(-3, 3, num=7),
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'early_stopping': [True, False],
        'validation_fraction': [0.05, 0.2]
    },
    'RandomForestClassifier': {
        'n_estimators': [10, 20, 40, 80, 160, 320],
        'min_samples_split': [2, 4, 8],
        'max_features': ['sqrt', 'log2'],
        'min_impurity_decrease': np.logspace(-5, 0, num=6)
    }
}

estimators = {
    'LogisticRegression': LogisticRegression(),
    'MLPClassifier': MLPClassifier(),
    'RandomForestClassifier': RandomForestClassifier()
}
