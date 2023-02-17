import numpy as np
import pandas as pd

import operator

from sklearn.base import BaseEstimator

# Points, which are closest to decision boundary are the most ambiguous one. TODO
def decision_boundary_most_ambiguous(clf, X_pool, unknown_indexes):
    ind = np.argmin(np.abs( 
        list(clf.decision_function(X_pool.iloc[unknown_indexes]) )
        ))
    return unknown_indexes[ind]

class UncertaintySampling():
    def least_confident(clf: BaseEstimator, X_pool: pd.DataFrame) -> pd.Index:
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
        return [X_pool.index[0]]

from sklearn.base import BaseEstimator

class QueryByCommittee():
    ''' Strategy attempting to minimize the version space. '''
    def __init__(X_train: pd.DataFrame, y_train: pd.DataFrame):
        QueryByCommittee.X_train = X_train
        QueryByCommittee.y_train = y_train

    def vote_entropy(X_pool):
        ''' Measures the level of disagreement between models in committee. '''
        def entropy(x):
            C = len(QueryByCommittee.committee)
            sum = 0
            for y_i in QueryByCommittee.y_train.unique():
                votes = 0
                for committee_member in QueryByCommittee.committee:
                    if committee_member.predict(x) == y_i:
                        votes += 1
                sum += - votes/C * np.log(votes/C)
            return sum        

        X_pool = X_pool.loc[:, ~X_pool.columns.isin(["vote_entropy"])]
        X_pool['vote_entropy'] = X_pool.apply(lambda x : entropy(x))
        X_pool.sort_values(by='vote_entropy', ascending=False, inplace=True)

        return [X_pool.index[0]]

    def average_kl_divergence(X_pool):
        ''' Measures the level of disagreement between models in committee. '''
        committee = QueryByCommittee.committee
        C = len(committee)
        sum = 0
        for c in range(C):
            # TODO: sum += 
            return sum/C

    def query(clf: BaseEstimator, X_pool: pd.DataFrame, n_models: int, disagreement_measure: str) -> pd.Index:
        ''' Gets most informative samples. '''
        if not hasattr(QueryByCommittee, 'committee'):
            if not hasattr(QueryByCommittee, 'X_train'):
                print("First you need to initialize class by __init__.")
                return
            # Create committee
            models_list = []
            for _ in range(n_models):
                new_model = clf.__class__().fit(QueryByCommittee.X_train, QueryByCommittee.y_train)
                models_list.append(new_model)
            QueryByCommittee.committee = models_list
            return
        if disagreement_measure == "vote_entropy":
            disagreement_measure_method = QueryByCommittee.vote_entropy
        elif disagreement_measure == "kl_divergence":
            disagreement_measure_method = QueryByCommittee.average_kl_divergence
        else:
            print("Unknown method of measuring committee disagreement.")
            return
        
        index = disagreement_measure_method(X_pool)
        return index
        

    def reset_committee():
        delattr(QueryByCommittee, 'committee')