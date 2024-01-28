"""
Random sampling for active learning.
"""

import numpy as np
from modAL.utils.data import modALinput
from modAL.models import ActiveLearner

def random_sampling(learner: ActiveLearner, X: modALinput, n_instances: int = 1, replace: bool = False, ) -> np.ndarray:
    """
    Random sampling query strategy.

    Args:
        X: The samples.
        n_instances: The number of instances to be sampled.
        replace: Whether the sample is with or without replacement.

    Returns:
        The indices of the instances from X chosen to be labelled.
    """

    n = len(X)
    assert n > 0, 'Found array with 0 sample(s), while a minimum of 1 is required'
    if n == 1:
        return [0], X
    
    if n_instances >= n:
        n_instances = n

    indices = np.random.choice(n, size=n_instances, replace=replace)
    values = X[indices]

    return indices, values
