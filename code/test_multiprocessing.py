# In order to measure Jupter Notebook's time type in the cell %%time
import time
import functools
import multiprocessing

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier as GBC

RANDOM_STATE = 13

breast_cancer = load_breast_cancer()

df_breast_cancer = pd.DataFrame(data= np.c_[breast_cancer['data'], breast_cancer['target']],
                     columns= list(breast_cancer['feature_names']) + ['target'])

X_train, X_test, y_train, y_test = train_test_split(df_breast_cancer.loc[:, df_breast_cancer.columns != 'target'], df_breast_cancer.loc[:, 'target'], test_size=0.2, random_state=RANDOM_STATE)

def measure_time(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        res = func(*args, **kwargs)
        stop_time = time.perf_counter()
        print(f"Finished {func.__name__!r} in {(stop_time-start_time):.4f} secs")
        return res
    return wrapper_timer

def debug(func):
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()] # the !r specifier means that repr() is used to represent the value.
        signature = ", ".join(args_repr + kwargs_repr)          
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")           
        return value
    return wrapper_debug

#@debug
#@measure_time
def train_model(Clf, X, y):
    model = Clf()
    model.fit(X, y)
    return model

def train_all_models(*args):
    _ = train_model(SVC, X_train, y_train)
    _ = train_model(GBC, X_train, y_train)
    return

NO_ITERATIONS = 100

@measure_time
def train_loop():
    for _ in range(NO_ITERATIONS):
        train_all_models()
        

@measure_time
def multiproc_train_loop():
    with multiprocessing.Pool() as pool:
        pool.map(train_all_models, range(NO_ITERATIONS)) # If iterable is large, then you should use chunking (by default it sets chunksize to best optimization value)


if __name__ == '__main__':
    print("Loop without multiprocessing took:")
    train_loop()

    print("Multiprocessing loop took:")
    multiproc_train_loop()

'''
TODO:
- global variable with synching across pool processes, which saves outcomes in any case (if there is error or keyboard interrupt)
'''

"""
Examples:
--------
>>> python test_multiprocessing.py
Loop without multiprocessing took:
Finished 'train_loop' in 26.5682 secs
Multiprocessing loop took:
Finished 'multiproc_train_loop' in 7.5309 secs
"""