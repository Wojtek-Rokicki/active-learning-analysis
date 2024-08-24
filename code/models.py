import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
DEFAULT_EPSILON = 0.1

class SGDLogClassifier(SGDClassifier):
    def __init__(self,
        *,
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        epsilon=DEFAULT_EPSILON,
        n_jobs=None,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False,
        average=False,
        ):
        super().__init__(
            loss="log_loss",
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            n_jobs=n_jobs,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            class_weight=class_weight,
            warm_start=warm_start,
            average=average,
        )

class SGDModifiedHuberClassifier(SGDClassifier):
    def __init__(self,
        *,
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        epsilon=DEFAULT_EPSILON,
        n_jobs=None,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False,
        average=False,
        ):
        super().__init__(
            loss="modified_huber", # could be replaced with "hinge". modified_huber is smoothed version. It cannot, because it does not provide probabilities. Smoothed version is equal to SVC smoothed.
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            n_jobs=n_jobs,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            class_weight=class_weight,
            warm_start=warm_start,
            average=average,
        )

class WeightedGaussianNB(GaussianNB):
    def fit(self, X, y, sample_weight=None):
        # Initialize the parameters
        self.classes_ = np.unique(y)
        self.theta_ = np.zeros((len(self.classes_), X.shape[1]))  # means
        self.sigma_ = np.zeros((len(self.classes_), X.shape[1]))  # variances
        self.class_prior_ = np.zeros(len(self.classes_))  # class priors
        self.class_count_ = np.zeros(len(self.classes_))
        
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        
        # Calculate the weighted means, variances, and class priors
        for i, y_i in enumerate(self.classes_):
            X_i = X[y == y_i]
            sw_i = sample_weight[y == y_i]
            total_weight = np.sum(sw_i)
            self.theta_[i, :] = np.average(X_i, axis=0, weights=sw_i)
            self.sigma_[i, :] = np.average((X_i - self.theta_[i, :]) ** 2, axis=0, weights=sw_i)
            self.class_prior_[i] = total_weight
            self.class_count_[i] = total_weight

        # Normalize the class priors
        self.class_prior_ /= np.sum(self.class_prior_)
        
        return self
    
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones_like(y)

        if not hasattr(self, "classes_"):
            self.classes_ = np.unique(y) if classes is None else classes
            self.theta_ = np.zeros((len(self.classes_), X.shape[1]))
            self.sigma_ = np.zeros((len(self.classes_), X.shape[1]))
            self.class_prior_ = np.zeros(len(self.classes_))
            self.class_count_ = np.zeros(len(self.classes_))
            self.n_features_ = X.shape[1]
        
        for i, y_i in enumerate(self.classes_):
            X_i = X[y == y_i]
            sw_i = sample_weight[y == y_i]
            total_weight = np.sum(sw_i)

            if total_weight == 0:
                continue
            
            old_count = self.class_count_[i]
            new_count = old_count + total_weight

            if old_count == 0:
                # Initialize if first batch for this class
                self.theta_[i, :] = np.average(X_i, axis=0, weights=sw_i)
                self.sigma_[i, :] = np.average((X_i - self.theta_[i, :]) ** 2, axis=0, weights=sw_i)
            else:
                # Update the mean and variance incrementally
                new_theta_i = np.average(X_i, axis=0, weights=sw_i)
                new_sigma_i = np.average((X_i - new_theta_i) ** 2, axis=0, weights=sw_i)

                self.theta_[i, :] = (self.theta_[i, :] * old_count + new_theta_i * total_weight) / new_count
                self.sigma_[i, :] = (
                    old_count * self.sigma_[i, :] + old_count * self.theta_[i, :] ** 2 +
                    total_weight * new_sigma_i + total_weight * new_theta_i ** 2
                ) / new_count - self.theta_[i, :] ** 2

            self.class_count_[i] = new_count

        # Update and normalize class priors
        self.class_prior_ = self.class_count_ / np.sum(self.class_count_)

        return self
    
    def _joint_log_likelihood(self, X):
        # Small constant to prevent division by zero
        epsilon = 1e-9

        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            # Add epsilon to the variance to prevent log(0) or log(neg_val)
            adjusted_sigma = np.maximum(self.sigma_[i, :], epsilon)
            jointi = np.log(self.class_prior_[i])
            n_ij = -0.5 * np.sum(np.log(2. * np.pi * adjusted_sigma))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) / adjusted_sigma, axis=1)
            joint_log_likelihood.append(jointi + n_ij)
        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood