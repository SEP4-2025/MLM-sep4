import os
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import Dict 
from . import schemas
from . import ml_model

# --- Configuration ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
# *** IMPORTANT: Make sure this filename matches your saved model ***
MODEL_FILENAME = "best_xgb.joblib" # <-- CHANGE THIS IF NEEDED
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# --- Lifespan Management (for model loading) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    print(f"Attempting to load model from: {MODEL_PATH}")
    app.state.model = None  # Use app.state to store shared resources

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        # The /predict endpoint will return 503 if model isn't loaded.
    else:
        try:
            app.state.model = ml_model.load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load model. Error: {e}")
            # Leave app.state.model as None so /predict errors out.

    yield  # Application runs from here

    # --- Shutdown ---
    print("Application shutting down.")
    app.state.model = None


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Simple ML Prediction API",
    description="API to serve a simple machine learning model via a POST endpoint.",
    version="0.1.0",
    lifespan=lifespan # Register the lifespan context manager
)


# --- API Endpoint ---

@app.post("/predict", response_model=schemas.OutputData)

async def post_predict(input_data: schemas.InputData):
    model = app.state.model
    if model is None:
        raise HTTPException(503, "Model is not loaded")

    try:
        # 1. extract only features for the model
        feature_payload = {
            "temperature": input_data.temperature,
            "light":       input_data.light,
            "airHumidity": input_data.airHumidity,
            "soilHumidity":input_data.soilHumidity
        }

        # 2. call your ML predict
        result = ml_model.predict(model=model, input_data=feature_payload)

        # 3. build the OutputData **using the full input_data** for echo
        return schemas.OutputData(
            prediction=       result["prediction"],
            prediction_proba= result.get("prediction_proba"),
            input_received=   input_data  # <-- here we pass the full Pydantic model
        )

    except ValueError as ve:
        raise HTTPException(400, f"Invalid input data: {ve}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Unexpected error during prediction: {e}")
@app.get(
    "/ping",
    tags=["Health"],
    summary="Simple health check endpoint"
)
async def ping():
    """
    Simple endpoint to check if the API is running.
    
    Returns:
        A dictionary with a status message.
    """
    return {"status": "ok", "message": "API is running"}