import subprocess
from typing import Union, List
import numpy as np
import json
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score, precision_recall_curve
from datetime import datetime

current_timestamp = datetime.now()
current_timestamp_str = current_timestamp.strftime("%d/%m/%Y, %H:%M:%S")


def compute_git_sha() -> str:   
    """
    Create ID for each git file. Keeping the data integrity

    Parameters
    ----------
    None
    
    Returns:
        str: git sha hash
    """

    try:
        # Return a hash for the current commit
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return sha
    except Exception:
        return "N/A"

def save_metrics(
    y_hat: Union[List, np.array, pd.DataFrame],
    y_true: Union[List, np.array, pd.DataFrame],
    output_path: str
) -> None:
    """
    Log relevant performance metrics for ML classification tasks.

    Save results in the given path

    Parameters
    ----------
    y_hat: Union[List, np.array, pd.DataFrame]
        Array with the model binary predictions {0, 1}

    y_true: Union[List, np.array, pd.DataFrame]
        Array with the ground truth values {0, 1}

    output_path: str
        Path to save the result metrics

    Returns:
        None:

    Example:
        >>> log_metrics([1, 0], [1, 1], "./logs.csv")
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_hat)

    metrics = {
        "r2": r2_score(y_true, y_hat),
        "roc_auc": roc_auc_score(y_true, y_hat),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "timestamp": current_timestamp_str,  # TODO: Include git SHA
        "git_sha": compute_git_sha(),
    }

    with open(output_path, "w") as f:
        json.dump(metrics, f)
