from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.staticfiles import StaticFiles
import numpy as np

app = FastAPI()

# Load the model
model = joblib.load('random_forest_model.joblib')

class InputData(BaseModel):
  features: list

@app.post("/predict")
async def predict(data: InputData):
  # Convert input data to numpy array
  input_data = np.array(data.features).reshape(1, -1)
  
  # Make prediction
  prediction = model.predict(input_data)
  
  return {"prediction": prediction[0]}

@app.get("/")
async def root():
  return {"message": "Welcome to the Census Income Predictor API"}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
  return FileResponse('static/index.html')