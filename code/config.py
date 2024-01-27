import pathlib

# Importing sklearn classificators
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
    
from modAL.random_sampling import random_sampling
from modAL.uncertainty import uncertainty_sampling
from modAL.expected_error import expected_error_with_loss, expected_error_reduction
from variance_reduction import fisher_information_sampling

# Saving results paths
FILES_DIRECTORY = pathlib.Path(__file__).parent.resolve()
BENCHMARKS_PATH =  FILES_DIRECTORY / '../data/benchmarks'
PLOTS_PATH = FILES_DIRECTORY / '../results/plots/'
PARTIAL_RESULTS_PATH = FILES_DIRECTORY / '../results/partial/'

# General
RANDOM_STATE_SEED = 13
N_KCV = 4
TEST_SIZE = 0.2 # equals to inversion of k parameter in k-fold CV
INITIAL_TRAIN_SIZE = 0.1 # fraction of train samples to be initial active learning train set

# Experiments flags
VERBOSE = 1
HYPERPARAMETERS_TUNNING = False
WEIGHTED_TRAINING = False
FULL_LEARNER_COMPARISON = True

# Datasets
DATASETS_NAMES = [          # ID    Repository & Target             Ratio     #S      #F
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
        # "webpage",          # 23    LIBSVM, w7a, target: minority   33:1      34,780  300  ## takes so long ...
        "ozone_level",      # 24    UCI, ozone, data                34:1      2,536   72
        "mammography",      # 25    UCI, target: minority           42:1      11,183  6
        # "protein_homo",     # 26    KDD CUP 2004, minority          11:1      145,751 74  ## takes so long ...
        "abalone_19"        # 27    UCI, target: 19                 130:1     4,177   10
    ] 
        #   "htru2"             # fin   UCI, target: minority           10:1      17,898  8


CLASSIFIERS = [
    (GaussianNB, {}),
    (LogisticRegression, {}),
    (SVC, {"probability": True}),
    (KNeighborsClassifier, {}),
    (RandomForestClassifier, {}),
    (GradientBoostingClassifier, {}),
]


ACTIVE_LEARNING_METHODS = {
    "random_sampling": {
        "params": {
            'query_strategy': random_sampling,
            'query_strategy_parameters': {}
        },
        "classifiers": CLASSIFIERS
    },
    "uncertainty_sampling": {
        "params": {
            'query_strategy': uncertainty_sampling, 
            'query_strategy_parameters': {}
        },
        "classifiers": CLASSIFIERS
    },
    "expected_error_reduction_01": {
        "params": {
            'query_strategy': expected_error_with_loss(expected_error_reduction, loss_type="binary"), 
            'query_strategy_parameters': {"pool_candidates_size": 250}
        },
        "classifiers": CLASSIFIERS
    },
    "expected_error_reduction_log": {
        "params": {
            'query_strategy': expected_error_with_loss(expected_error_reduction, loss_type="log"), 
            'query_strategy_parameters': {"pool_candidates_size": 250}
        },
        "classifiers": CLASSIFIERS
    },
    "variance_reduction": {
        "params": {
            'query_strategy': fisher_information_sampling, 
            'query_strategy_parameters': {"pool_candidates_size": 250}
        },
        "classifiers": CLASSIFIERS

    }
    # TODO: Add other methods
}

COMPUTATIONALLY_COMPLEX_METHODS = [
    'expected_error_reduction',
    'variance_reduction'
]