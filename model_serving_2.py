"""
FastAPI - przygotowanie szkieletu API pod serving
- endpoint /healthcheck (GET)
- endpoint /predict (POST)
"""

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI()

# odpalenie
# python -m fastapi dev model_serving_2.py

class InputText(BaseModel):
    texts: list[str]

class ModelOutput(BaseModel):
    results: list[str]

@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}

@app.post("/predict")
def predict(inputs: InputText) -> ModelOutput:
    results = []
    
    for text in inputs.texts:
        # Później tutaj wstawimy model
        results.append(
            random.choice(["pozytywny",
                            "negatywny", 
                            "neutralny"])
        )
    return ModelOutput(results=results)
