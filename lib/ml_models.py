"""
This module contains the Machine Learning models that have been evaluated and the differente parametrization options.
"""

from . import config

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier

random_state = config.my_random_state

models = {
   "LogisticRegression": LogisticRegression,
    "KNeighborsClassifier": KNeighborsClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "AdaBoostClassifier": AdaBoostClassifier,
    "XGBClassifier":XGBClassifier
}

model_parameter_rules = {
    LogisticRegression:[
            {
                "C": [0.01, 0.1, 1, 10], 
                "penalty": ["l1","l2",None], 
                "solver": ["saga"], 
                "max_iter": [100, 1000, 10000, 50000],
                "random_state": [random_state]
            },
            {
                "C": [0.01, 0.1, 1, 10], 
                "penalty": ["elasticnet"],
                "solver": ["saga"], 
                "max_iter": [100, 1000, 10000, 50000],
                "l1_ratio": [0,1],                
                "random_state": [random_state]
            },        
            {
                "C": [0.01, 0.1, 1, 10], 
                "penalty": ["l2",None], 
                "solver": ["sag"], 
                "max_iter": [100, 1000, 10000, 50000],
                #"l1_ratio": [0,1,None],                
                "random_state": [random_state]
            },        
            {
                "C": [1, 10, 100, 1000, 2000], 
                "penalty": [None, "l2"], 
                "solver": ["lbfgs"],
                "max_iter": [100, 2000, 10000, 20000],
                "random_state": [random_state]
            }
    ],
    KNeighborsClassifier: [
            {
                "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
                "weights": ["uniform", "distance"],
                "algorithm": ["brute"],
                "metric": [
                    "minkowski",
                ],
                "p": [1, 1.5, 2]
            },
    ],
    RandomForestClassifier: [
            {
                "n_estimators": [50, 75, 100, 150, 200], 
                "criterion": ["gini","entropy", "log_loss"], 
                "max_features": ["sqrt", "log2"],
                "class_weight": ["balanced"],
                "min_samples_leaf": [1, 2, 3, 4],
                "min_samples_split": [1, 2, 3, 4],
                "max_depth": [None],
                "random_state": [random_state]
            }
    ],
    DecisionTreeClassifier: [
            {
                "criterion": ["gini","entropy", "log_loss"], 
                "splitter": ["best", "random"],
                "max_features": ["sqrt", "log2", None],
                "class_weight": ["balanced"],
                "min_samples_leaf": [1, 2, 3, 4],
                "min_samples_split": [1, 2, 3, 4],
                "max_depth": [None, 5, 10, 15, 20],
                "random_state": [random_state]
            }
    ],
    AdaBoostClassifier: [
            {
                "n_estimators": [50, 75, 100, 150, 200],
                "learning_rate": [0.1, 0.5, 1, 5, 10],
                "random_state": [random_state]
            }
    ],
    XGBClassifier:[
            {
                "n_estimators": [50, 75, 100, 150, 200],
                "grow_policy": ["depthwise", "lossguide"],
                "learning_rate": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                "max_depth": [4, 5, 6, 7, 8, 9],
                "device": ["gpu"],
                "random_state": [random_state]
            }
    ]
}
