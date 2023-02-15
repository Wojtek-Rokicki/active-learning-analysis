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
    def least_confident(clf: BaseEstimator, X_pool: pd.DataFrame, k: int = 1) -> pd.Index:
        """ Selects most informative samples.

        Selects indices of the most informative samples for
        model - samples, which most probable label is least 
        probable among other samples.

        Parameters
        ----------
        clf : BaseEstimator
            model which has predict_proba method
        X_pool : array-like
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
        X_pool = X_pool.loc[:, ~X_pool.columns.isin(["pred_proba", "max_proba"])]
        X_pool['pred_proba'] = X_pool.apply(lambda x : dict(
            zip(clf.classes_, clf.predict_proba(x.values[None])[0])
            ), axis = 1)
        X_pool['max_proba'] = X_pool['pred_proba'].apply(lambda x: sorted(x.items(), key=operator.itemgetter(1), reverse=True)[0])
        X_pool.sort_values(by='max_proba', ascending=True, key=lambda col: col.map(lambda x: x[1]), inplace=True) # key should expect Series
        indices = X_pool.iloc[:k].index # use loc to index pd.DataFrame by Index object, drop for removing
        return indices

from sklearn.base import BaseEstimator

class QueryByCommittee():
    ''' Strategy attempting to minimize the version space. '''
    def __init__(model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.DataFrame, X_pool: pd.DataFrame, n_models: int = 2):
        model = model # issubclass(model, BaseEstimator) == True
        X_pool = X_pool
        # Create committee
        models_list = []
        for _ in range(n_models):
            new_model = model.__class__().fit(X_train, y_train)
            models_list.append(new_model)

    def vote_entropy():
        ''' Measures the level of disagreement between models in committee. '''

        pass

    def average_kl_divergence():
        ''' Measures the level of disagreement between models in committee. '''
        pass

    def query(method: str, ):
        ''' Gets most informative samples. '''
        if method == "vote_entropy":
            method = vote_entropy
        elif method == "kl_divergence": # TODO: Check if there is significant difference in methods for binary classification
            method = average_kl_divergence
        else:
            print("Unknown method of measuring committee disagreement.")
            return
        
        pass