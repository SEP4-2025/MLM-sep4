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
    temperature: float = Field(
        ...,
        description="Current temperature reading for the greenhouse"
    )
    light: float = Field(
        ...,
        description="Current light level reading for the greenhouse"
    )
    airHumidity: float = Field(
        ...,
        description="Current air humidity reading for the greenhouse"
    )
    soilHumidity: float = Field(
        ...,
        description="Current soil humidity reading for the greenhouse"
    )
    date: datetime = Field(
        ...,
        description="Timestamp when this full set of sensor readings was taken"
    )
    greenhouseId: int = Field(
        ...,
        description="Identifier of the greenhouse to which these readings belong"
    )
    sensorReadingId: int = Field(
        ...,
        description="Database ID of the individual sensor reading event"
    )

    class Config:
        schema_extra = {
            "example": {
                "temperature": 23.5,
                "light": 400.0,
                "airHumidity": 48.2,
                "soilHumidity": 75.0,
                "date": "2025-05-19T10:15:00",
                "greenhouseId": 2,
                "sensorReadingId": 1234
            }
        }

# --- Output Schema ---
class OutputData(BaseModel):
    # The model's predicted next-hour soil humidity value
    prediction: float = Field(
        ...,
        description="Predicted soil humidity percentage for the next hour"
    )
    # Optional probabilities for each class (if the model supports predict_proba)
    prediction_proba: Optional[List[float]] = Field(
        None,
        description="Optional list of class probabilities from the model"
    )
    # Echo back the input that produced this prediction
    input_received: InputData = Field(
        ...,
        description="The input payload that was used to generate this prediction"
    )

    class Config:
        schema_extra = {
            "example": {
                "prediction": 74.32,
                "prediction_proba": [0.8, 0.2],
                "input_received": {
                    "temperature": 23.65,
                    "light": 391.35,
                    "airHumidity": 48.46,
                    "soilHumidity": 75.00,
                    "date": "2025-05-06T10:37:25.760000",
                    "greenhouseId": 1,
                    "sensorReadingId": 4321
                }
            }
        }