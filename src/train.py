# TODO: Implement training script.
# CLI: python -m src.train --data data/customer_churn_synth.csv --outdir artifacts/

from abc import ABC
import argparse  # Read CLI params
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import r2_score, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_val_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
OUTPUT_VAR = "churned"
MODELS = {"logistic_reg": LogisticRegressionCV(), "xgboost": XGBClassifier(), "random_forest": RandomForestClassifier(), "lgb": LGBMClassifier()}

class MLModelTrainer(ABC):
    def __init__(self, data_path, output_dir, model = "logistic_reg"):
        self.data_path: str = data_path  # TODO: Add try except
        self.input_data: pd.DataFrame = pd.read_csv(self.data_path)
        self.output_dir: str = output_dir
        self.target_var: str = OUTPUT_VAR
        self.model = MODELS[model]  # Initialize Classification Model
        self.arrays: tuple = None
        self.input_cols = [col for col in self.input_data if col != self.target_var]
        self.ouput_cols = [col for col in self.input_data if col == self.target_var]
        self.metrics = {}
        self.scaler = None 

    def split_data(self) -> None:
        X = self.input_data[self.input_cols]
        y = self.input_data[self.ouput_cols]

        X_train, X_val, y_train, y_val = train_val_split(
            X, y, val_size=0.33, random_state=RANDOM_SEED
        )

        self.arrays = {
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
        }

    def save_artifacts(self) -> None:
        try:
            assert self.model != None
        except:
            raise AssertionError(
                f"There is no ML model trained, please try to run .train() before saving artifacts."
            )
        # TODO: save model.pkl, feature_pipeline.pkl, feature_importances.csv 
    
    def preprocess_data(self):
        X_train,  X_val,  y_train,  y_val = self.arrays
        
        # Process categorical features
        X_train["contract_type"] = X_train["contract_type"].map({"Monthly": 0, "Annual": 1})
        X_train["autopay"] = X_train["autopay"].map({"No": 0, "Yes": 1})
        X_train["is_promo_user"] = X_train["is_promo_user"].map({"No": 0, "Yes": 1})

        # Scale all variables
        self.scaler = StandardScaler()
        self.scaler.fit_transform(X_train)
        self.scaler.transform(X_val)  # Just to avoid data leakage

        self.arrays = (X_train,  X_val,  y_train,  y_val)
    
    def hpo(self):
        pass
        
    def log_metrics(self) -> None:
        X_train,  X_val,  y_train,  y_val = self.arrays
        y_hat = self.model.predict(X_val, y_val)

        return {
            "r2": r2_score(y_val, y_hat),
            "roc_auc":  roc_auc_score(y_val, y_hat),
            "roc_pr":  precision_recall_curve(y_val, y_hat),
        }


class ChurnModelTrainer(MLModelTrainer):
    def __init__(self):
        super().__init__()

    def train(self):
        X_train,  X_val,  y_train,  y_val = self.arrays
        self.model.fit(X_train, y_train)


def main() -> None:
    """
    Description.

    Parameters
    ----------
    arg1 : type
        Description
    arg2 : type
        Description
    arg3 : type
        Description

    Returns:
        type:

    Example:
        >>> ('arg1', 'arg2')
        'output'
    """

    # Read CLI params
    parser = argparse.ArgumentParser(description="Training Pipeline")
    args = parser.parse_args()
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file")
    data, output_dir = args.data, args.outdir

    # Initialize ML Personalized Class
    model_trainer = ChurnModelTrainer(data, output_dir)

    # Run Training Pipeline
    model_trainer.split_data()
    model_trainer.preprocess_data()
    model_trainer.train()
    model_trainer.log_metrics()
    model_trainer.save_artifacts()

    return None


if __name__ == "__main__":
    main()
