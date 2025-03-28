"""
FastAPI - przygotowanie szkieletu API pod serving
- endpoint /healthcheck (GET)
- endpoint /predict (POST)
"""

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# zadanko - healthcheck endpoint
# - ma zwracac {"status": "ok"} gdy się go wywoła


# zadanko - predict endpoint
# - ma przyjmowac liste ciagow znakow (wejscie do modelu)
# -> zwracac liste ciagow znakow (wynik predykcji modelu)
#   input i output ma miec zdefiniowane typy pydantic
#   na razie niech dopisuje " ml-workout" do kazdego ciagu znakow
