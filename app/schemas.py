from pydantic import BaseModel

class FertilizerInput(BaseModel):
    Soil_pH: float
    Soil_Moisture: float
    Organic_Carbon: float
    Electrical_Conductivity: float
    Nitrogen_Level: int
    Phosphorus_Level: int
    Potassium_Level: int
    Temperature: float
    Humidity: float
    Rainfall: float
    Fertilizer_Used_Last_Season: float
    Yield_Last_Season: float
    Soil_Type: str
    Crop_Type: str
    Crop_Growth_Stage: str
    Season: str
    Irrigation_Type: str
    Previous_Crop: str
    Region: str
