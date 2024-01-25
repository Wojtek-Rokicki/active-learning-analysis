from sklearn.base import BaseEstimator
from modAL.utils.data import modALinput

from modAL.utils.selection import (multi_argmin, shuffled_argmin)

from numpy.linalg import LinAlgError
import numpy as np

# @change_repr
def fisher_information_sampling(classifier: BaseEstimator, X: modALinput,
                    n_instances: int = 1, random_tie_break: bool = False, pool_candidates_size: int = None,
                    **uncertainty_measure_kwargs) -> np.ndarray:
    """
    Fisher Information sampling query strategy. Selects the instances which influences the sensitivity of the model's likelihood function with respect to the model parameters least.
    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
            measure function.
    Returns:
        The indices of the instances from X chosen to be labelled.
        The margin metric of the chosen instances.
    """

    if pool_candidates_size != None and pool_candidates_size <= X.shape[0]:
        indices = np.random.choice(X.shape[0], size=pool_candidates_size, replace=False)
        X = X[indices]

    probabilities = classifier.predict_proba(X, **uncertainty_measure_kwargs)
    W_x = np.diag(np.prod(probabilities, axis=1))
    I_u = np.linalg.multi_dot([X.T, W_x, X])

    fi = []
    for i in range(W_x.shape[0]):
        I_x = np.outer((X[i]*W_x[i,i]), X[i])
        try:
            inv_I_x = np.linalg.inv(I_x)
        except LinAlgError:
            inv_I_x = np.linalg.pinv(I_x) # or other options such as: dropping the sample (singularity means low sensitivity of the model to that sample); regularization (adding small positive values to the diagonal of the initial matrix to stabilize the calculations); pseudo-inverse (this situation - it exists for all matrices)
        tr = np.trace(np.linalg.multi_dot([I_u, inv_I_x]))
        fi.append(tr)

    fi = np.array(fi)

    if not random_tie_break:
        return multi_argmin(fi, n_instances=n_instances)

    return shuffled_argmin(fi, n_instances=n_instances)