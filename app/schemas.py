from pydantic import BaseModel, Field
from typing import List, Optional, Union
from enum import Enum
from datetime import datetime

# Define enum for sensor types
class SensorType(str, Enum):
    AIR_TEMPERATURE = "TEMPERATURE"
    AIR_HUMIDITY = "HUMIDITY"
    LIGHT = "LIGHT"
    SOIL_HUMIDITY = "SOIL"

# Map sensor IDs to their types
class SensorId(int, Enum):
    AIR_TEMPERATURE_SENSOR = 1
    AIR_HUMIDITY_SENSOR = 2
    LIGHT_SENSOR = 3
    SOIL_HUMIDITY_SENSOR = 4

# --- Input Schema ---
class InputData(BaseModel):
    id: int = Field(..., description="Reading ID")
    value: float = Field(..., description="Sensor reading value")
    timeStamp: datetime = Field(..., description="Timestamp of the reading")
    sensorId: SensorId = Field(..., description="ID of the sensor")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "value": 22,
                "timeStamp": "2025-05-01T08:00:00",
                "sensorId": 1
            }
        }

# --- Output Schema ---
class OutputData(BaseModel):
    # Example prediction output - Adjust based on what your model predicts
    prediction: Union[float, str, List[float]] = Field(..., description="The prediction result from the model (e.g., a score, class label, or list of probabilities)")
    prediction_proba: Optional[List[float]] = Field(None, description="Optional: Prediction probabilities if applicable")
    input_received: InputData = Field(..., description="A copy of the input data that was processed")
    # Add other relevant information if needed
    # model_version: Optional[str] = Field(None, description="Version of the model used")