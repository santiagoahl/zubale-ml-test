# CLI: python -m src.train --data data/customer_churn_synth.csv --outdir artifacts/

# SHAP for feature importance
import shap
from shap import KernelExplainer, Explainer

# Data and CLI management
import os
import json
import joblib
import pickle as pkl
import argparse 
import pandas as pd
import matplotlib.pyplot as plt

# ML
from abc import ABC  # Abstract Classes
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline  # Inference inference_pipeline
from sklearn.compose import ColumnTransformer  # feature inference_pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from .metrics import save_metrics

# Config vars
RANDOM_SEED = 42
OUTPUT_VAR = "churned"
MODELS = {
    "logistic_reg": LogisticRegressionCV(),
    "xgboost": XGBClassifier(),
    "random_forest": RandomForestClassifier(),
    "lgb": LGBMClassifier(),
}

#  Abstract Class for ML Pipeline tasks (split, feature pipeline, inference, train, etc.) 
class MLClassifier(ABC):
    def __init__(self):
        pass

    def split_data(self) -> None:

        X = self.input_data[self.input_cols]
        y = self.input_data[self.ouput_cols]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.33, random_state=RANDOM_SEED
        )

        self.arrays = {
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train.values.ravel(),
            "y_val": y_val.values.ravel(),
        }

    def save_artifacts(self) -> None:
        try:
            assert self.model is not None
        except:
            raise AssertionError(
                "There is no ML model trained, please try to run .train() before saving artifacts."
            )

        X_val = self.arrays["X_val"]

        # 1. Save Model and feature_pipeline
        joblib.dump(self.model, self.artifact_paths["model"])
        joblib.dump(self.feature_pipeline, self.artifact_paths["feature_pipeline"])

        # 2. Compute and save feature importance with SHAP
        X_val_sample = X_val.iloc[: self.shap_n_samples]
        X_val_enc = pd.DataFrame(
            self.inference_pipeline.named_steps["features"].transform(X_val_sample),
            columns=self.inference_pipeline.named_steps[
                "features"
            ].get_feature_names_out(),
        )

        self.shap_explainer = Explainer(
            self.inference_pipeline.named_steps["model"], X_val_enc
        )
        self.shap_values = self.shap_explainer(X_val_enc)

        shap_df = pd.DataFrame(self.shap_values.values, columns=X_val_enc.columns)
        shap_df.to_csv(self.artifact_paths["feature_importances"], index=False)

        shap.summary_plot(self.shap_values.values, X_val_enc, show=False)

        fig = plt.gcf()
        fig.suptitle("SHAP Summary Plot - Customer Churn Model", fontsize=16, y=1.02)
        plot_path = os.path.join(self.output_dir, "shap_summary_plot.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def preprocess_data(self) -> None:
        X_train = self.arrays["X_train"]
        X_val = self.arrays["X_val"]
        y_train = self.arrays["y_train"]
        y_val = self.arrays["y_val"]

        self.cat_features = [
            "plan_type",
            "contract_type",
            "autopay",
            "is_promo_user",
        ]
        self.num_features = [
            col for col in self.input_cols if col not in self.cat_features
        ]

        self.feature_pipeline = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features),
                ("num", self.scaler, self.num_features),
            ]
        )

    def hpo(self) -> None:
        pass

    def log_metrics(self) -> None:
        X_train = self.arrays["X_train"]
        X_val = self.arrays["X_val"]
        y_train = self.arrays["y_train"]
        y_val = self.arrays["y_val"]
        y_hat = self.inference_pipeline.predict(X_val)

        # Compute performance metrics on val
        save_metrics(y_hat, y_val, self.artifact_paths["metrics"])


#  Subclass for customer churn use case
class ChurnModelTrainer(MLClassifier):
    def __init__(self, data_path, output_dir, model="logistic_reg"):
        self.data_path: str = data_path  # TODO: Add try except
        self.input_data: pd.DataFrame = pd.read_csv(self.data_path)
        self.output_dir: str = output_dir
        self.target_var: str = OUTPUT_VAR
        self.model = MODELS[model]  # Initialize Classification Model
        self.arrays: dict = None
        self.input_cols = [col for col in self.input_data if col != self.target_var]
        self.ouput_cols = [col for col in self.input_data if col == self.target_var]
        self.metrics = {}
        self.scaler = StandardScaler()
        self.feature_pipeline = None
        self.inference_pipeline = None
        self.artifact_paths = {
            "model": "model.pkl",
            "feature_pipeline": "feature_pipeline.pkl",
            "feature_importances": "feature_importances.csv",
            "metrics": "metrics.json",
        }
        self.artifact_paths = {
            key: os.path.join(self.output_dir, value)
            for key, value in self.artifact_paths.items()
        }
        self.features: list = []
        self.cat_features: list = []
        self.num_features: list = []
        self.shap_explainer = None
        self.shap_values = None
        self.shap_n_samples = 100
        self.feature_pipeline = None
        self.inference_pipeline = None

    def train(self):
        X_train = self.arrays["X_train"]
        X_val = self.arrays["X_val"]
        y_train = self.arrays["y_train"]
        y_val = self.arrays["y_val"]

        self.feature_pipeline.fit(X_train)
        self.inference_pipeline = Pipeline(
            [
                ("features", self.feature_pipeline),
                ("model", self.model),
            ]
        )
        self.inference_pipeline.fit(X_train, y_train)


def main() -> None:
    """
    Run Training Inference Pipeline
    """
    # Read CLI params
    parser = argparse.ArgumentParser(description="Train Churn Model")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--outdir", type=str, required=True, help="Path to CSV file")
    args = parser.parse_args()
    data, output_dir = (args.data, args.outdir)

    # Initialize ML Personalized Class for churn
    model_trainer = ChurnModelTrainer(data, output_dir)

    # Run Training inference_pipeline
    model_trainer.split_data()
    model_trainer.preprocess_data()
    model_trainer.train()
    model_trainer.log_metrics()  
    model_trainer.save_artifacts()


if __name__ == "__main__":
    main()