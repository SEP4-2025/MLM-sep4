def load_model(model_path):
    """
    Stub function for loading a model.
    In this version, we don't actually load a model, just return a placeholder.
    """
    print(f"Note: No model loaded from {model_path} (placeholder only)")
    return "DUMMY_MODEL"

def predict(model, input_data):
    """
    Stub function for prediction.
    Returns dummy data that matches the OutputData schema.
    """
    return {
        "prediction": 42.0,
        "prediction_proba": [0.9, 0.1],
        "input_received": input_data
    }