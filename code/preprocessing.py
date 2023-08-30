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

# Dataset spliting
from sklearn.model_selection import train_test_split
from collections import Counter

def al_split(X_raw, y_raw, test_size, initial_train_size, random_state, verbose=False):
    # Train and test split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=test_size, random_state=random_state, shuffle=True, stratify=y_raw)

    # Initial train and pool split
    X_train_init, X_pool, y_train_init, y_pool = train_test_split(X_train, y_train, train_size=initial_train_size, random_state=random_state, shuffle=True, stratify=y_train)

    if verbose:
        print("Train dataset:")
        print(f"{len(y_train)} samples")
        print(f"{Counter(y_train)}")

        print("Test dataset:")
        print(f"{len(y_test)} samples")
        print(f"{Counter(y_test)}")

        print("Init train dataset:")
        print(f"{len(y_train_init)} samples")
        print(f"{Counter(y_train_init)}")

        print("Pool dataset:")
        print(f"{len(y_pool)} samples")
        print(f"{Counter(y_pool)}")

    return X_train, X_train_init, X_pool, X_test, y_train, y_train_init, y_pool, y_test 