import numpy as np
import pandas as pd

import operator

# Points, which are closest to decision boundary are the most ambiguous one. TODO
def decision_boundary_most_ambiguous(clf, X_pool, unknown_indexes):
    ind = np.argmin(np.abs( 
        list(clf.decision_function(X_pool.iloc[unknown_indexes]) )
        ))
    return unknown_indexes[ind]

class UncertaintySampling():
    def least_confident(clf, X: pd.DataFrame, k=1):
        X['pred_proba'] = X.apply(lambda x : dict(
            zip(clf.classes_, clf.predict_proba(x.values[None])[0])
            ), axis = 1)
        X['max_proba'] = X['pred_proba'].apply(lambda x: sorted(x.items(), key=operator.itemgetter(1), reverse=True)[0])
        X.sort_values(by='max_proba', ascending=True, key=lambda col: col.map(lambda x: x[1]), inplace=True) # key should expect Series
        return X.iloc[:k].index # use loc to index pd.DataFrame by Index object, drop for removing