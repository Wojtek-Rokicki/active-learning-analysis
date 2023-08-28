# Standardize dataframe
import pandas as pd
from sklearn.preprocessing import StandardScaler

def split_dataset_xy(dataset: pd.DataFrame, target_name="target") -> list[pd.DataFrame]:
    X = dataset.loc[:, dataset.columns != "target"]#.to_numpy()
    y = dataset["target"]#.values
    return X, y

def merge_dataset_xy(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    X["target"] = y
    return X

def standardize(dataset: pd.DataFrame) -> list[pd.DataFrame]:
    X, y = split_dataset_xy(dataset)
    scaler = StandardScaler().fit(X)
    X_standardized = scaler.transform(X)
    return merge_dataset_xy(pd.DataFrame(X_standardized), y)