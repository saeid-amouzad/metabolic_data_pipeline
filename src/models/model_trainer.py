"""
ModelTrainer

- Load feature-selected data
- Train multiple models with CV
- Select best model by metric
- Retrain on full dataset
- Save model artifact
- Save validated metadata for inference

"""

import json
import logging
from pathlib import Path
from datetime import datetime

import joblib
import yaml
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ======================================================
# ModelTrainer
# ======================================================
class ModelTrainer:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.data_path = Path(self.config["paths"]["features"])
        self.artifact_dir = Path(self.config["paths"]["artifacts"])
        self.metadata_path = Path(self.config["paths"]["metadata"])

        self.target = self.config["modeling"]["target_col"]
        self.selection_metric = self.config["modeling"]["metric"]

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        self._setup_logger()

        self.logger.info("Artifacts dir: %s", self.artifact_dir.resolve())
        self.logger.info("Metadata path: %s", self.metadata_path.resolve())

    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ModelTrainer")

    # --------------------------------------------------
    # Data loading
    # --------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)

        assert not df.empty, "Feature CSV is empty"
        assert self.target in df.columns, "Target column missing"

        return df

    # --------------------------------------------------
    # Model registry
    # --------------------------------------------------
    def get_models(self):
        return {
            "logistic_regression": (
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", LogisticRegression(max_iter=1000)),
                    ]
                ),
                {"model__C": [0.1, 1.0, 10]},
            ),
            "random_forest": (
                RandomForestClassifier(random_state=42),
                {"n_estimators": [100, 200], "max_depth": [None, 10]},
            ),
        }

    # --------------------------------------------------
    # Training & model selection
    # --------------------------------------------------
    def train_select_best(self):
        df = self.load_data()

        X = df.drop(columns=[self.target])
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )

        best_model = None
        best_name = None
        best_score = -1
        best_params = None
        best_metrics = None

        for name, (model, param_grid) in self.get_models().items():
            self.logger.info("Training %s", name)

            search = GridSearchCV(
                model,
                param_grid=param_grid,
                scoring=self.selection_metric,
                cv=3,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)

            y_prob = search.best_estimator_.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "threshold": 0.5,
            }

            self.logger.info("Evaluation metrics: %s", metrics)
            self.logger.info("%s ROC-AUC: %.3f", name, metrics["roc_auc"])

            self._save_predictions(y_test, y_pred, y_prob)

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_model = search.best_estimator_
                best_name = name
                best_params = search.best_params_
                best_metrics = metrics

        assert best_model is not None, "No model was selected"
        assert best_metrics is not None, "Best metrics not captured"

        # --------------------------------------------------
        # Retrain best model on FULL dataset
        # --------------------------------------------------
        self.logger.info("Retraining best model: %s", best_name)
        best_model.fit(X, y)

        # --------------------------------------------------
        # Save best trained model and metadata
        # --------------------------------------------------
        model_path = self._save_model(best_model, best_name)
        metadata_path = self._save_metadata(
            model_name=best_name,
            score=best_score,
            params=best_params,
            feature_names=list(X.columns),
            metrics=best_metrics,
        )

        # Hard validation
        self._validate_metadata(metadata_path)

        self.logger.info("✅ Model saved to: %s", model_path)
        self.logger.info("✅ Metadata saved to: %s", metadata_path)

        return best_model

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------
    def _save_model(self, model, name):
        path = self.artifact_dir / f"{name}_final.joblib"
        joblib.dump(model, path)
        return path

    def _save_metadata(self, model_name, score, params, feature_names, metrics):
        metadata = {
            "model_name": model_name,
            "trained_at": datetime.utcnow().isoformat(),
            "selection_metric": self.selection_metric,
            "best_score": float(score),
            "best_params": params,
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "metrics": metrics,
        }

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return self.metadata_path

    # --------------------------------------------------
    # Metadata validation (for inference safety)
    # --------------------------------------------------
    def _validate_metadata(self, path: Path):
        with open(path) as f:
            metadata = json.load(f)

        required_keys = {
            "model_name",
            "trained_at",
            "selection_metric",
            "best_score",
            "best_params",
            "n_features",
            "feature_names",
            "metrics",
        }

        missing = required_keys - set(metadata.keys())
        assert not missing, f"Metadata missing keys: {missing}"

        assert isinstance(metadata["feature_names"], list)
        assert metadata["n_features"] == len(metadata["feature_names"])
        assert "roc_auc" in metadata["metrics"]

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------
    def _save_predictions(self, y_true, y_pred, y_prob):
        pd.DataFrame({"y_true": y_true}).to_csv(
            self.artifact_dir / "y_true.csv", index=False
        )
        pd.DataFrame({"y_pred": y_pred}).to_csv(
            self.artifact_dir / "y_pred.csv", index=False
        )
        pd.DataFrame({"y_prob": y_prob}).to_csv(
            self.artifact_dir / "y_prob.csv", index=False
        )

        self.logger.info("Saved test predictions for diagnostics")


# ======================================================
# Stand-alone example
# ======================================================
if __name__ == "__main__":
    """
    Stand-alone training run.

    Usage:
        python src/models/model_trainer.py
    """
    trainer = ModelTrainer("config/config.yaml")
    trainer.train_select_best()
    print("Model training completed!")
