from typing import List, Type
from sklearn.compose import ColumnTransformer  # feature inference_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def build_feature_pipeline(
    encoder: Type[OneHotEncoder],
    scaler: Type[StandardScaler],
    categorical_features: List[str],
    numerical_features: List[str],
) -> str:
    """
    Initialize column transformer for a given encoder (cat to num), scaler, as well as categoriacal and numerical features.

    Parameters
    ----------
    encoder: type
        Description
    scaler: type
        Description
    categorical_features: List[str]
        Description
    numerical_features: List[str]
        Description

    Returns:
        type:

    Example:
        >>> ('arg1', 'arg2')
        'output'
    """
    return ColumnTransformer(
        [
            ("cat", encoder, categorical_features),
            ("num", scaler, numerical_features),
        ]
    )