# Implement PSI (for categorical variables) and KS test (for continuous ones) for data shift triggering
# CLI: python -m src.drift --ref data/churn_ref_sample.csv --new data/churn_shifted_sample.csv

import argparse
import numpy as np
import json
import pandas as pd
import pandas.api.types as ptypes  # Split features intro cat and num
from scipy.stats import ks_2samp
from typing import TypedDict
import os

ARTIFACTS_DIR = "artifacts/"
DRIFT_REPORT_FILENAME = "drift_report.json"
DRIFT_REPORT_PATH = os.path.join(ARTIFACTS_DIR, DRIFT_REPORT_FILENAME)


# Default values
class DriftData(TypedDict, total = False):
    threshold: float  # If the summarized drift probs exceeed, trigger drift
    overall_drift: bool
    features: dict[str, float]  # Features with its drift probability



class DriftComputer():
    # TODO: disengage methods applying strategy pattern
    def __init__(self, feature_name, df_ref, df_new):
        self.feature_name = feature_name
        self.df_ref = df_ref
        self.df_new = df_new
        self.feature_type = "num" if ptypes.is_numeric_dtype(df_ref[feature_name]) else "cat"

    @staticmethod    
    def compute_ks_test(feature_name: str, df_ref: pd.DataFrame, df_new: pd.DataFrame) -> float:
        """
        Run Kolmogorov-Smirnov test for two samples

        Parameters
        ----------
        feature_name: str
            Variable name to perform the test
        df_ref: pd.DataFrame 
            Reference Data
        df_new: pd.DataFrame
            New Data

        Returns:
            float: Probability of Data Shift for the given feature

        Example:
            >>> compute_ks_test('downtime_hours_30d', pd.read_csv("churn_ref_sample.csv"), pd.read_csv("churn_shifted_sample.csv"))
            'output'
        """
        feature_values_ref = df_ref[feature_name]
        feature_values_new = df_new[feature_name]
        ks_statistic, p_value = ks_2samp(feature_values_ref, feature_values_new)
        drift_prob = 1 - p_value
        return drift_prob 

    @staticmethod
    def compute_psi_test(feature_name: str, df_ref: pd.DataFrame, df_new: pd.DataFrame, bins: int = None) -> float:
        """
        Run Population Stability Index test for two samples (categorical or binned numerical)

        Parameters
        ----------
        feature_name: str
            Name of the feature/column to perform the PSI test on. Can be categorical or numerical 
            (if numerical, bins can optionally be applied before computing PSI).
        df_ref: pd.DataFrame 
            Reference dataset (baseline data, e.g., training set)
        df_new: pd.DataFrame
            New dataset to compare against the reference (e.g., production or recent sample)
        bins: int, optional
            Number of bins to use for numerical features. If None, each unique value is treated as a category.

        Returns
        -------
        float
            PSI value indicating the degree of data shift for the given feature.
            Higher values indicate larger distributional drift.

        Example
        -------
            >>> compute_psi_test('plan_type', pd.read_csv("churn_ref_sample.csv"), pd.read_csv("churn_shifted_sample.csv"))
            0.12
        """
        # Define distance thresholds: (min_psi, max_psi) triggers drift

        min_psi = 0.1
        max_psi = 0.25
        
        # GEt data   
        ref = df_ref[feature_name]
        new = df_new[feature_name]

        # If numeric and no bins given, treat unique values as categories
        if ptypes.is_numeric_dtype(ref) and bins is not None:
            ref = pd.cut(ref, bins=bins)
            new = pd.cut(new, bins=bins)

        # Compute normalized value counts
        ref_dist = ref.value_counts(normalize=True)
        new_dist = new.value_counts(normalize=True)

        # Align all categories/bins
        all_categories = set(ref_dist.index).union(new_dist.index)
        ref_dist = ref_dist.reindex(all_categories, fill_value=0)
        new_dist = new_dist.reindex(all_categories, fill_value=0)

        # Compute PSI
        psi_value = np.sum((ref_dist - new_dist) * np.log((ref_dist + 1e-6) / (new_dist + 1e-6)))

        if psi_value <= min_psi:
            return np.clip(psi_value / max_psi, 0.0, 1.0)
        if psi_value >= max_psi:
            return 1.0
        
        # Rescale to 0-1
        psi_value_scaled = (psi_value - min_psi) / (max_psi - min_psi)
        psi_value_scaled = np.clip(psi_value_scaled / max_psi, 0.0, 1.0)
        return psi_value_scaled

    def compute_prob(self):
        if self.feature_type == "num":
            return self.compute_ks_test(self.feature_name, self.df_ref, self.df_new)
        else:
            return self.compute_psi_test(self.feature_name, self.df_ref, self.df_new, bins=10)
        

def monitor_drift(data_ref_path: str, data_new_path: str) -> None:
    """
    Run data shift test for all feature variables.

    Parameters
    ----------
    data_ref_path : str
        Path to the Reference Data
    data_new_path : str
        Path to the New Data

    Returns:
        None: Results are saved in the data directory

    Example:
        >>> monitor_drift("data/churn_ref_sample.csv", "data/churn_shifted_sample.csv")
    """
    # Read Data and features
    # TODO: validate if both dfs have the same features (try-except)

    df_ref = pd.read_csv(data_ref_path)
    df_new = pd.read_csv(data_new_path)
    features = list(set(df_ref.columns.union(df_new.columns)))

    features_num = [col for col in df_ref.columns 
                if ptypes.is_numeric_dtype(df_ref[col])]

    features_cat = [col for col in df_ref.columns 
                if ptypes.is_object_dtype(df_ref[col]) or ptypes.is_string_dtype(df_ref[col])]

    # Run Kolmogorov-Smirnoff Tests and save results
    drift_data: DriftData = DriftData(threshold=0.2, overall_drift=False, features={})

    drift_data["features"] = {
        feature_name: DriftComputer(feature_name, df_ref, df_new).compute_prob()
        for feature_name in features
    }

    # Compute overall drift (simple average over the ks probs)
    threshold = drift_data["threshold"]
    prob_drift_overall = np.max(
        [prob_not_h0 for feature, prob_not_h0 in drift_data["features"].items()]
    )  # Pragmatic approach: If any feature exceed the threshold -> neither classical avg nor weighted one do
    if prob_drift_overall > threshold:
        drift_data["overall_drift"] = True

    # Save Drift Report
    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(drift_data, f, indent=4)


def monitor_drift_cli():
    """
    Compute KS tests for data shift
    """
    # Read CLI params: path to input data and artifacts dir
    parser = argparse.ArgumentParser(description="Train Churn Model")
    parser.add_argument(
        "--ref", type=str, required=True, help="Path to previous input data (CSV)"
    )
    parser.add_argument(
        "--new", type=str, required=True, help="Path to new input data (CSV)"
    )
    args = parser.parse_args()
    data_ref_path, data_new_path = (args.ref, args.new)

    # Run Monitoring
    monitor_drift(data_ref_path, data_new_path)


if __name__=="__main__":
    monitor_drift_cli()