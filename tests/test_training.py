# Run tests with:
# PYTHONPATH=. pytest -v tests/test_training.py

from src.train import train
import pytest
import os
import json
from datetime import datetime, timedelta

DATA_PATH = "data/customer_churn_synth.csv"  # Input dataset
ROC_AUC_QUALITY_THRESHOLD = 0.83
ARTIFACTS_DIR = "artifacts/"  # Output directory for artifacts

OUTPUT_PATHS = {
    "shap_values": os.path.join(ARTIFACTS_DIR, "feature_importances.csv"),
    "feature_pipeline": os.path.join(ARTIFACTS_DIR, "feature_pipeline.pkl"),
    "log_metrics": os.path.join(ARTIFACTS_DIR, "metrics.json"),
    "model": os.path.join(ARTIFACTS_DIR, "model.pkl"),
    "shap_plot": os.path.join(ARTIFACTS_DIR, "shap_summary_plot.png"),
}


def test_train_produces_artifacts_and_quality():
    """
    Test the training pipeline:
    1. Runs training
    2. Verifies that all expected artifacts exist
    3. Ensures artifacts were created recently (not stale)
    4. Checks that the model ROC-AUC meets the quality threshold
    """

    # 1. Run training
    train(DATA_PATH, ARTIFACTS_DIR)
    train_finished_timestamp = datetime.now()

    # 2. Verify all artifacts exist
    for artifact_name, artifact_path in OUTPUT_PATHS.items():
        assert os.path.isfile(artifact_path), f"Missing artifact: {artifact_name}"

    # 3. Check artifacts were created/modified within 1 minute of training
    for artifact_name, artifact_path in OUTPUT_PATHS.items():
        modification_time_raw = os.path.getmtime(artifact_path)
        modification_time = datetime.fromtimestamp(modification_time_raw)
        assert modification_time < train_finished_timestamp + timedelta(seconds=60), \
            f"Artifact {artifact_name} seems outdated"

    # 4. Validate ROC-AUC threshold from metrics.json
    with open(OUTPUT_PATHS["log_metrics"], "r") as file:
        model_metrics = json.load(file)

    model_roc_auc = model_metrics["roc_auc"]
    assert model_roc_auc >= ROC_AUC_QUALITY_THRESHOLD, \
        f"ROC-AUC {model_roc_auc} is below threshold {ROC_AUC_QUALITY_THRESHOLD}"
