from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from pydantic.types import conlist
from fastapi.middleware.cors import CORSMiddleware
import joblib

model = joblib.load("./xgboost_redshift_model.pkl")

class InputData(BaseModel):
    values: conlist(float, min_length=5, max_length=5)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontnd url ["http://frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(data: InputData):
    try:
        prediction = model.predict([data.values])
        result = float(prediction[0]) 
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
