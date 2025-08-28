# run test with PYTHONPATH=. pytest -v tests/test_training.py
from src.train import train
import os
import json
from datetime import datetime, timedelta


DATA_PATH = "data/customer_churn_synth.csv" # data
ROC_AUC_QUALITY_THRESHOLD = 0.83
ARTIFACTS_DIR = "artifacts/"  # out_dir

OUTPUT_PATHS = {
    "shap_values": os.path.join(ARTIFACTS_DIR, "feature_importances.csv"),
    "feature_pipeline": os.path.join(ARTIFACTS_DIR, "feature_pipeline.pkl"),
    "log_metrics": os.path.join(ARTIFACTS_DIR, "metrics.json"),
    "model": os.path.join(ARTIFACTS_DIR, "model.pkl"),
    "shap_plot":  os.path.join(ARTIFACTS_DIR, "shap_summary_plot.png"),
}


def test_train() -> None:
    # Run training

    train(DATA_PATH, ARTIFACTS_DIR)
    train_finished_timestamp = datetime.now()

    # Validate there exist artifacts
    for artifact_name, artifact_path in OUTPUT_PATHS.items():
        assert os.path.isfile(artifact_path)

    # Validate the artifacts are saved within a few time ago
    # To avoid checking old files instead of current ones 
    for artifact_name, artifact_path in OUTPUT_PATHS.items():
        modification_time_raw = os.path.getmtime(artifact_path)  # When was the last edition time for each artifact 
        modification_time = datetime.fromtimestamp(modification_time_raw)  # human readable
        assert modification_time < train_finished_timestamp + timedelta(seconds=60)   # Verify the file was edited within 1 minute after the inference pipeline ran

    # Validate ROC-AUC threshold
    with open(OUTPUT_PATHS["log_metrics"], "r") as file:
        model_metrics = json.load(file)
    
    model_roc_auc = model_metrics["roc_auc"]

    assert model_roc_auc >= ROC_AUC_QUALITY_THRESHOLD


if __name__=="__main__":
    test_train()