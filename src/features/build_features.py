import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def get_training_testing_data(
        df: pd.DataFrame,
        target: str,
        test_size: float = 0.2,
        scale_features: bool = False,
        random_state: int = 42
        ):
    if scale_features:
        df['age'] = scale_feature(df['age'])
        df['hours_studied'] = scale_feature(df['hours_studied'])
        
    X = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def get_prediction_data(
    df: pd.DataFrame,
    target: str,
    scale_features: bool = False,
    ):
    if scale_features:
        df['age'] = scale_feature(df['age'])
        df['hours_studied'] = scale_feature(df['hours_studied'])
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y


def scale_feature(feature: np.ndarray, scaler:str='normalize'):
    """scales a feature to be centered around 0 and have a standard deviation of 1

    Args:
        feature (np.ndarray): feature to be scaled
        scaler (str, optional): scaler to be used (normalize or max_min). Defaults to 'normalize'.

    Returns:
        np.ndarray: scaled feature
    """
    SCALERS = ['normalize', 'max_min']
    assert scaler in SCALERS, f'scaler must be one of {SCALERS}'
    if scaler == 'normalize':
        return StandardScaler().fit_transform(feature)
    return MinMaxScaler().fit_transform(feature)
    