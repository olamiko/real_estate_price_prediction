import pickle
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any
import numpy as np

app = FastAPI(title="housing-price-prediction")

with open('models/pipeline.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

@app.post("/predict")
def predict(house_detail: Dict[str, Any]):
    result = np.expm1(pipeline.predict(house_detail))
    return {"prediction": float(result[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
