# Importing sklearn classificators
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
    
from modAL.uncertainty import uncertainty_sampling
from modAL.disagreement import vote_entropy_sampling, consensus_entropy_sampling, max_disagreement_sampling
from modAL.expected_error import expected_error_reduction
from variance_reduction import fisher_information_sampling

# Saving results paths
BENCHMARK_PATH = './data/benchmark'
MODELS_PATH = './results/models/'
PLOTS_PATH = './results/plots/'
PARTIAL_RESULTS_PATH = './results/partial/'

# General
RANDOM_STATE_SEED = 13
TEST_SIZE = 0.2
INITIAL_TRAIN_SIZE = 0.1

# Experiments flags
VERBOSE = 1
FULL_LEARNER_COMPARISON = True

# Datasets
DATASETS = [      # ID    Repository & Target             Ratio     #S      #F
        "ecoli",            # 1     UCI, target: imU                8.6:1     336     7
        "optical_digits",   # 2     UCI, target: 8                  9.1:1     5,620   64
        "satimage",         # 3     UCI, target: 4                  9.3:1     6,435   36
        "pen_digits",       # 4     UCI, target: 5                  9.4:1     10,992  16
        "abalone",          # 5     UCI, target: 7                  9.7:1     4,177   10
        "sick_euthyroid",   # 6     UCI, target: sick euthyroid     9.8:1     3,163   42
        "spectrometer",     # 7     UCI, target: >=44               11:1      531     93
        "car_eval_34",      # 8     UCI, target: good, v good       12:1      1,728   21
        "isolet",           # 9     UCI, target: A, B               12:1      7,797   617
        "us_crime",         # 10    UCI, target: >POOL_SIZE5        12:1      1,994   100
        "yeast_ml8",        # 11    LIBSVM, target: 8               13:1      2,417   103
        "scene",            # 12    LIBSVM, target: >one label      13:1      2,407   294
        "libras_move",      # 13    UCI, target: 1                  14:1      360     90
        "thyroid_sick",     # 14    UCI, target: sick               15:1      3,772   52
        "coil_2000",        # 15    KDD, CoIL, target: minority     16:1      9,822   85
        "arrhythmia",       # 16    UCI, target: 06                 17:1      452     278
        "solar_flare_m0",   # 17    UCI, target: M->0               19:1      1,389   32
        "oil",              # 18    UCI, target: minority           22:1      937     49
        "car_eval_4",       # 19    UCI, target: vgood              26:1      1,728   21
        "wine_quality",     # 20    UCI, wine, target: <=4          26:1      4,898   11
        "letter_img",       # 21    UCI, target: Z                  26:1      20,000  16
        "yeast_me2",        # 22    UCI, target: ME2                28:1      1,484   8
        "webpage",          # 23    LIBSVM, w7a, target: minority   33:1      34,780  300
        "ozone_level",      # 24    UCI, ozone, data                34:1      2,536   72
        "mammography",      # 25    UCI, target: minority           42:1      11,183  6
        "protein_homo",     # 26    KDD CUP 2004, minority          11:1      145,751 74
        "abalone_19"        # 27    UCI, target: 19                 130:1     4,177   10
    ] 

# Classifiers with hyperparameters tunning
GS_CLASSIFIERS = {
    "GNB": {"Clf": GaussianNB, "params": "gs"},
    "LR": {"Clf": LogisticRegression, "params": "gs"},
    "SVC": {"Clf": SVC, "params": "gs"},
    "KNN": {"Clf": KNeighborsClassifier, "params": "gs"},
    "RFC": {"Clf": RandomForestClassifier, "params": "gs"},
    "GBC": {"Clf": GradientBoostingClassifier, "params": "gs"}
}

PARAMS_GRID = {
    "GaussianNB": {
        "var_smoothing": [1, 1e-3, 1e-6, 1e-9]
    },
    "LogisticRegression": [
      {
        "class_weight": [None, 'balanced'],
        "solver": ['lbfgs'],
        "penalty": ['l2', None],
        "C": [100, 10, 1.0, 0.1, 0.01]
    },
    {
        "class_weight": [None, 'balanced'],
        "solver": ['liblinear'],
        "penalty": ['l1', 'l2'],
        "C": [100, 10, 1.0, 0.1, 0.01]
    }
    ],
    "KNeighborsClassifier": {
        "n_neighbors": [2, 3, 5, 10, 15],
        "leaf_size": [20, 25, 30, 35],
        "p": [1, 2], 
    },
    "SVC": {
        "class_weight": [None, 'balanced'],
        "gamma": ['auto', 'scale'],
        "kernel": ['linear', 'rbf'],
        "C": [100, 10, 1.0, 0.1],
        "probability": [True],
        "random_state": [13]   
    },
    "RandomForestClassifier": {
        "class_weight": [None, 'balanced'],
        "n_estimators": [25, 100, 250, 500],
        "min_samples_split": [3, 7, 10, 13],
        "max_depth": [None, 2, 3, 5]  
    },
    "GradientBoostingClassifier": {
        "n_estimators": [25, 100, 250, 500],
        "min_samples_split": [3, 7, 10, 13],
        "max_depth": [1, 2, 3, 5]  
    }
}

# Active learning configurations

# Stopping criterion configuration
from enum import Enum
class StoppingCriterion(Enum):
  N_QUERIES = 1
  FRACTION_OF_POOL_QUERIES = 2
  ENTROPY_CONFIDENCE = 3
  # TODO: Other criterions

# Querying configuration
AL_QUERY_BATCH_SIZE = 1

# N Queries configuration
N_QUERIES = 50 # fixed number of queries

# Fraction of pool queries
FRACTION_OF_POOL_QUERIES = 0.25 # TODO: INITIAL_TRAIN_SIZE + FRACTION_OF_POOL_QUERIES <= 1.0 !

# Entropy Confidence configuration
N_DECLINE_ROUNDS = 5

# Query-By-Committee configuration
# Default hyperparameters for all models are GS tunned
COMMITTEES = {
    "GNB": {"committee_type": "homogenous", "Clf": GaussianNB, "n_models": 3},
    "LR": {"committee_type": "homogenous", "Clf": LogisticRegression, "n_models": 3},
    "SVC": {"committee_type": "homogenous", "Clf": SVC, "n_models": 3},
    "KNN": {"committee_type": "homogenous", "Clf": KNeighborsClassifier, "n_models": 3},
    "RFC": {"committee_type": "homogenous", "Clf": RandomForestClassifier, "n_models": 3},
    "GBC": {"committee_type": "homogenous", "Clf": GradientBoostingClassifier, "n_models": 3},
    "GNB": {"committee_type": "homogenous", "Clf": GaussianNB, "n_models": 6},
    "LR": {"committee_type": "homogenous", "Clf": LogisticRegression, "n_models": 6},
    "SVC": {"committee_type": "homogenous", "Clf": SVC, "n_models": 6},
    "KNN": {"committee_type": "homogenous", "Clf": KNeighborsClassifier, "n_models": 6},
    "RFC": {"committee_type": "homogenous", "Clf": RandomForestClassifier, "n_models": 6},
    "GBC": {"committee_type": "homogenous", "Clf": GradientBoostingClassifier, "n_models": 6},
    "GNB": {"committee_type": "homogenous", "Clf": GaussianNB, "n_models": 9},
    "LR": {"committee_type": "homogenous", "Clf": LogisticRegression, "n_models": 9},
    "SVC": {"committee_type": "homogenous", "Clf": SVC, "n_models": 9},
    "KNN": {"committee_type": "homogenous", "Clf": KNeighborsClassifier, "n_models": 9},
    "RFC": {"committee_type": "homogenous", "Clf": RandomForestClassifier, "n_models": 9},
    "GBC": {"committee_type": "homogenous", "Clf": GradientBoostingClassifier, "n_models": 9},
    "LR, SVC, KNN": {"committee_type": "heterogenous", "Clfs": [LogisticRegression, SVC, KNeighborsClassifier]},
    "LR, SVC, RFC": {"committee_type": "heterogenous", "Clfs": [LogisticRegression, SVC, RandomForestClassifier]},
    "LR, SVC, GBC": {"committee_type": "heterogenous", "Clfs": [LogisticRegression, SVC, GradientBoostingClassifier]},
    "SVC, KNN, GBC": {"committee_type": "heterogenous", "Clfs": [LogisticRegression, SVC, GradientBoostingClassifier]},
    "SVC, KNN, RFC": {"committee_type": "heterogenous", "Clfs": [LogisticRegression, SVC, GradientBoostingClassifier]},
    "LR, KNN, GBC": {"committee_type": "heterogenous", "Clfs": [LogisticRegression, SVC, GradientBoostingClassifier]},
    "LR, KNN, RFC": {"committee_type": "heterogenous", "Clfs": [LogisticRegression, SVC, GradientBoostingClassifier]},
}

# Variance Reduction configuration
CUSTOM_VR_CLASSIFIERS = {
  "LR": {
    "Clf": LogisticRegression, 
    "params": {"penalty": None, "dual": False, "tol": 0.0001, "C": 1.0, "fit_intercept": True, "intercept_scaling": 1, "class_weight": None, "random_state": RANDOM_STATE_SEED, "solver": 'lbfgs', "max_iter": 100, "multi_class": 'auto', "verbose": 0, "warm_start": False, "n_jobs": None, "l1_ratio": None}
    }
}

# Expected Error Reduction configuration
CUSTOM_EER_CLASSIFIERS = {
  "GNB": {
    "Clf": GaussianNB,
    "params": "gs"
  }
}

active_learning_methods = {
    "uncertainty_sampling": {
        "params": [{'query_strategy': uncertainty_sampling, 'stopping_criterion': StoppingCriterion.FRACTION_OF_POOL_QUERIES}],
        "classifiers": GS_CLASSIFIERS
    },
    "query_by_committee":{
        "params": [{'query_strategy': vote_entropy_sampling, 'stopping_criterion': StoppingCriterion.FRACTION_OF_POOL_QUERIES},
                    {'query_strategy': consensus_entropy_sampling, 'stopping_criterion': StoppingCriterion.FRACTION_OF_POOL_QUERIES},
                    {'query_strategy': max_disagreement_sampling, 'stopping_criterion': StoppingCriterion.FRACTION_OF_POOL_QUERIES}],
        "classifiers": COMMITTEES,

    },
    "query_by_bagging":{
        "params": [{'query_strategy': vote_entropy_sampling, 'stopping_criterion': StoppingCriterion.FRACTION_OF_POOL_QUERIES},
                    {'query_strategy': consensus_entropy_sampling, 'stopping_criterion': StoppingCriterion.FRACTION_OF_POOL_QUERIES},
                    {'query_strategy': max_disagreement_sampling, 'stopping_criterion': StoppingCriterion.FRACTION_OF_POOL_QUERIES}],
        "classifiers": COMMITTEES,
    },
    # "query_by_boosting", TODO
    # "expected_model_change", TODO
    "expected_error_reduction": {
        "params": [{'query_strategy': expected_error_reduction, 'stopping_criterion': StoppingCriterion.FRACTION_OF_POOL_QUERIES}],
        "classifiers": CUSTOM_EER_CLASSIFIERS
    },
    "variance_reduction": {
        "params":[{'query_strategy': fisher_information_sampling, 'stopping_criterion': StoppingCriterion.FRACTION_OF_POOL_QUERIES}],
        "classifiers": CUSTOM_VR_CLASSIFIERS

    }
    # "density_weighted" TODO
}