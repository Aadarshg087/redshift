from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, TypeAdapter
from typing import List
from pydantic.types import conlist
import joblib

model = joblib.load("./xgboost_redshift_model.pkl")

class InputData(BaseModel):
    values: conlist(float, min_length=5, max_length=5)

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    try:
        prediction = model.predict([data.values])
        # Convertin numpy.float32 to standard python float
        result = float(prediction[0])
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# health check
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}