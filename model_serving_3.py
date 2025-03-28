"""
Model Serving - wczytanie modelu
- i logika w funkcji predict
"""

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# zadanie - /predict ma
#   1. wczytac zapisany model
#   2. odpalic go na danych wejsciowych - i zwrocic predykcje
