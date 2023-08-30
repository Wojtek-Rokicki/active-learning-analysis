# Hyperparameters tunning of the initial model
from sklearn.base import ClassifierMixin
from sklearn.model_selection._search import BaseSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, StratifiedKFold

def tune_classifiers_hyperparameters(Clf: ClassifierMixin, X, y, param_grid, GridSearchMethod: BaseSearchCV,  scorer, random_state=13, verbose=True) -> tuple:

    skf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
    # Any GridSearch assumes stratified folds if there is classifier
    if issubclass(GridSearchMethod, GridSearchCV):
        gs = GridSearchMethod(estimator=Clf(), param_grid=param_grid, scoring=scorer, refit=False, cv=skf)
    elif issubclass(GridSearchMethod, HalvingGridSearchCV):
        gs = GridSearchMethod(estimator=Clf(), param_grid=param_grid, scoring=scorer, refit=False, cv=skf, factor=2)
    else:
        raise ValueError("Unknown or unsupported BaseSearchCV") 
    
    gs.fit(X, y)

    if verbose:
        params_combinations = gs.cv_results_["params"]
        params_combinations_scores = gs.cv_results_["mean_test_score"]
        for i, params in enumerate(params_combinations):
            print(f"{params}\n{params_combinations_scores[i]}")

    return gs.best_params_