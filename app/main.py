import os
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import Dict 
from . import schemas
from . import ml_model

# --- Configuration ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
# *** IMPORTANT: Make sure this filename matches your saved model ***
MODEL_FILENAME = "your_model.pkl" # <-- CHANGE THIS IF NEEDED
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# --- Lifespan Management (for model loading) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    print(f"Attempting to load model from: {MODEL_PATH}")
    app.state.model = None # Use app.state to store shared resources
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        # The /predict endpoint will fail if the model isn't loaded.
    else:
        try:
            app.state.model = ml_model.load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load model. Error: {e}")
            # Model loading failed, app.state.model remains None

    yield # The application runs while yielded

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

@app.post(
    "/predict",
    response_model=schemas.OutputData,
    tags=["Prediction"],
    summary="Make Prediction based on Sensor Reading"
)
async def post_predict(input_data: schemas.InputData):
    """
    Endpoint to make predictions using the loaded ML model based on sensor data.

    - Accepts input data conforming to the **InputData** schema (including sensor ID, value, timestamp).
    - Returns predictions conforming to the **OutputData** schema.
    - Raises HTTP errors if the model is not loaded or prediction fails.
    """
    model = app.state.model
    if model is None:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="Model is not loaded. Cannot make predictions."
        )

    try:
        # Call the prediction function from ml_model module
        # This function needs to handle the InputData schema correctly
        prediction_output = ml_model.predict(model=model, input_data=input_data)
        return prediction_output

    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data for prediction: {ve}"
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"ERROR during prediction endpoint: {e}") 
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred during prediction: {e}"
        )

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