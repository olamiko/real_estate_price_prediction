import pickle
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any
import numpy as np

app = FastAPI(title="housing-price-prediction")

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

@app.post("/predict")
def predict(house_detail: Dict[str, Any]):
    return np.expm1(pipeline.predict(house_detail))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)