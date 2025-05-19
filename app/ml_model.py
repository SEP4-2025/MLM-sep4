import joblib
import numpy as np
from pathlib import Path
from typing import Any, Dict

# Path to your serialized model; adjust if e nevoie
MODEL_FILENAME = "best_xgb.joblib"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME


def load_model(model_path):
    """
    Load the trained ML model from disk.
    If no path is provided, load from default MODEL_PATH.
    """
    path = model_path or MODEL_PATH
    # Raise an error if model file is missing
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    # Deserialize the model object
    model = joblib.load(path)
    return model



def predict(model, input_data):
     """
    Generate prediction and (if available) probabilities for a single input.
    
    Parameters:
      - model: a scikit-learn estimator (or similar)
      - input_data: dict with keys matching your feature schema
    
    Returns:
      A dict with:
        - "prediction": float or str
        - "prediction_proba": list[float] or None
        - "input_received": the original input_data
    """
    # 1. Convert input_data dict to feature array (example)
    #    Adjust order/columns to match your training data!
     features = [
        input_data["temperature"],
        input_data["airHumidity"],
        input_data["light"],
        input_data["soilHumidity"]
        # ... any other fields
    ]
     X = np.array(features).reshape(1, -1)

    # 2. Make prediction
     raw_pred = model.predict(X)[0]
    # 3. Optionally get class probabilities
     pred_proba = None
     if hasattr(model, "predict_proba"):
        pred_proba = model.predict_proba(X)[0].tolist()

     return {
        "prediction": float(raw_pred),
        "prediction_proba": pred_proba,
        "input_received": input_data
    }