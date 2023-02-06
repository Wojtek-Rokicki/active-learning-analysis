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
        """ Selects most informative samples.

        Selects indices of the most informative samples for
        model - samples, which most probable label is least 
        probable among other samples.

        Parameters
        ----------
        clf : BaseEstimator
            model which has predict_proba method
        X : array-like
            pool DataFrame
        k : int, default=1
            how many most informative samples indices should 
            be returned

        Returns
        -------
        indices : pd.Index, length=k
            <output description>

        Examples
        --------
        >>> <execution instruction>
        <result for doctest>
        """
        X = X.loc[:, ~X.columns.isin(["pred_proba", "max_proba"])]
        X['pred_proba'] = X.apply(lambda x : dict(
            zip(clf.classes_, clf.predict_proba(x.values[None])[0])
            ), axis = 1)
        X['max_proba'] = X['pred_proba'].apply(lambda x: sorted(x.items(), key=operator.itemgetter(1), reverse=True)[0])
        X.sort_values(by='max_proba', ascending=True, key=lambda col: col.map(lambda x: x[1]), inplace=True) # key should expect Series
        indices = X.iloc[:k].index # use loc to index pd.DataFrame by Index object, drop for removing
        return indices

    # there is margin sampling, which takes into account the rest of the labels distribution
    # it takes samples, for which two most probable classes are indistinguishable
    # it comes down to least confident in binary classification

    # entropy - information needed to encode classes distribution. In binary problem it reduces to
    # above methods

# Query by Commitee (QBC)
# - Disagreement measurements
#   - Vote entropy
#   - Kullback-Leibler Divergence