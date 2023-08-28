# Importing sklearn classificators
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Import active learning classes and methods
import os
import sys
module_path = os.path.abspath(os.path.join('./active_learning/modAL/'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from modAL.disagreement import vote_entropy_sampling, consensus_entropy_sampling, max_disagreement_sampling, max_std_sampling

# Classificators configurations
def get_classificator(m):
    def __repr__(self):
        return self.__name__
    models_switcher={
        "GNB": {
            "model": GaussianNB,
            "params": [{}]
        },
        "LR": {
            "model": LogisticRegression,
            "params": [{}]
        },
        "SVC": {
            "model": SVC,
            "params": [{'C': 1.0, 'kernel': 'rbf', 'class_weight': None, 'max_iter': 1000, 'random_state': None, 'probability': True, 'tol': 0.0001, 'verbose': 0}]
        },
        "KNN": {
            "model": KNeighborsClassifier,
            "params": [{}]
        },
        "RFC": {
            "model": RandomForestClassifier,
            "params": [{}]
        },
        "GBC": {
            "model": GradientBoostingClassifier,
            "params": [{}]
        }
    }
    return models_switcher.get(m, "Invalid model")
    
classificator_names=[
    "GNB",
    "LR", 
    "SVC",
    "KNN",
    "RFC", 
    "GBC"
]

# Active learning configurations
POOL_SIZE = 0.6
MAX_ITERATIONS = 10
N_MODELS = 5

def get_active_learning_method(m):
    active_learning_methods_switcher={
        "uncertainty_sampling": {
            "params": [{'query_strategy': uncertainty_sampling, 'pool_size': POOL_SIZE, 'max_iterations': MAX_ITERATIONS},
                       {'query_strategy': uncertainty_sampling, 'pool_size': POOL_SIZE, 'max_iterations': MAX_ITERATIONS},
                       {'query_strategy': uncertainty_sampling, 'pool_size': POOL_SIZE, 'max_iterations': MAX_ITERATIONS}]
        },
        "query_by_committee":{
            "params":[{'n_models': N_MODELS, 'disagreement_measure': vote_entropy_sampling, 'pool_size': POOL_SIZE, 'max_iterations': MAX_ITERATIONS},
                      {'n_models': N_MODELS, 'disagreement_measure': consensus_entropy_sampling, 'pool_size': POOL_SIZE, 'max_iterations': MAX_ITERATIONS},
                      {'n_models': N_MODELS, 'disagreement_measure': max_disagreement_sampling, 'pool_size': POOL_SIZE, 'max_iterations': MAX_ITERATIONS}]

        },
        "expected_error_reduction": {},
        "variance_reduction": {}
    }
    return active_learning_methods_switcher.get(m, "Invalid method")


active_learning_methods=[
    "uncertainty_sampling",
    "query_by_committee",
    "variance_reduction"
]

param_grid = {
    "GNB": {
        "var_smoothing": [1, 1e-3, 1e-6, 1e-9]
    },
    "LR": {
        "class_weight": [None, 'balanced'],
        "solver": ['lbfgs', 'liblinear'],
        "penalty": ['l1', 'l2', 'elasticnet'],
        "C": [100, 10, 1.0, 0.1, 0.01]
    },
    "KNN": {
        "n_neighbors": [2, 3, 5, 10, 15],
        "leaf_size": [20, 25, 30, 35],
        "p": [1, 2], 
    },
    "SVC": {
        "class_weight": [None, 'balanced'],
        "gamma": ['auto', 'scale'],
        "kernel": ['linear', 'rbf'],
        "C": [100, 10, 1.0, 0.1],
        "probability": [True]    
    },
    "RFC": {
        "class_weight": [None, 'balanced'],
        "n_estimators": [25, 100, 250, 500],
        "min_samples_split": [3, 7, 10, 13],
        "max_depth": [None, 2, 3, 5]  
    },
    "GBC": {
        "n_estimators": [25, 100, 250, 500],
        "min_samples_split": [3, 7, 10, 13],
        "max_depth": [1, 2, 3, 5]  
    }
}