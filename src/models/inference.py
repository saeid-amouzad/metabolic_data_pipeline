"""
Loads the trained model artifact and provides a lightweight
prediction interface used by the Flask backend.

Online inference ONLY (NO training).

"""

import json
import joblib
import pandas as pd
from pathlib import Path

class InferenceService:
    """
    Online inference service.
    """

    def __init__(self, artifact_dir="src/models/artifacts"):
        self.artifact_dir = Path(artifact_dir)

        # --------------------------------------------------
        # Load metadata
        # --------------------------------------------------
        metadata_path = self.artifact_dir / "model_metadata.json"
        assert metadata_path.exists(), "Model metadata not found"

        with open(metadata_path) as f:
            metadata = json.load(f)

        # REQUIRED by ModelTrainer contract
        assert "model_name" in metadata, "metadata missing 'model_name'"
        assert "feature_names" in metadata, "metadata missing 'feature_names'"

        self.model_name = metadata["model_name"]
        self.feature_names = metadata["feature_names"]

        # --------------------------------------------------
        # Load trained model
        # --------------------------------------------------
        model_path = self.artifact_dir / f"{self.model_name}_final.joblib"
        assert model_path.exists(), f"Model artifact not found: {model_path}"

        self.model = joblib.load(model_path)

    # --------------------------------------------------
    # Predict probability of diabetes (y = 1)
    # --------------------------------------------------
    def predict(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame), "X must be a DataFrame"
        assert len(X) > 0, "Empty input"

        # Enforce feature order & presence
        X = X[self.feature_names]

        return self.model.predict_proba(X)[:, 1]


# --------------------------------------------------
# Stand-alone example
# --------------------------------------------------
if __name__ == "__main__":
    inference = InferenceService()

    X = (
        pd.read_csv("data/processed/features.csv")
        .drop(columns=["diabetes"])
        .iloc[[0]]
    )

    prob = inference.predict(X)[0]
    print(f"Predicted Type 2 diabetes risk: {prob:.3f}")
