"""
Model Serving - wczytanie modelu TYLKO RAZ
- i logika w funkcji predict
"""

from contextlib import asynccontextmanager
from typing import List


from fastapi import FastAPI
from pydantic import BaseModel

MODELS = {}


# wyjaśnimy po co to :)
@asynccontextmanager
async def lifespan(app):
    # load model on start
    print("Loading model")
    MODELS["sentiment_classifier"] = load_model("model_df.joblib")
    yield
    print("Cleaning up")
    MODELS.clear()


app = FastAPI(lifespan=lifespan)


# zadanie - dostosowac predict - zeby wczytywac model tylko raz
