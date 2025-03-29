"""
Intro do FastAPI
- poznanie podstaw FastAPI
- endpointy GET i POST
"""
from fastapi import FastAPI
from pydantic import BaseModel

# create FastAPI app
app = FastAPI()

# GET endpoint - hello world
@app.get("/hello_world")
def hello_world():
    return {"message": "Hello World"}

@app.post("/hello_world_name")
def hello_world_name(name, name2):
    return {"message": f"Hello {name} {name2}"}

@app.post("/add_numbers")
def add_numbers(number1: int, number2: int):
    result = number1 + number2
    return {"result": result}

@app.post("/add_numbers_list")
def add_numbers_list(numbers: list[int]):
    # result = 0
    # for number in numbers:
    #     result += number
    result = sum(numbers)
    return {"result": result}

# from pydantic import BaseModel
class InputDataNumbers(BaseModel):
    numbers: list[int]

class OutputDataNumbers(BaseModel):
    result: int

@app.post("/add_numbers_list2")
def add_numbers_list2(inputs: InputDataNumbers) -> OutputDataNumbers:
    result = sum(inputs.numbers)
    # wczesniej bylo
    # return {"result": result}
    return OutputDataNumbers(result=result)

class InputText(BaseModel):
    texts: list[str]

class OutputText(BaseModel):
    result: str

@app.post("/concatenate_text")
def concatenate_text(inputs: InputText) -> OutputText:
    # todo - połączyć texty " "
    # result = ""
    # for text in inputs.texts:
    #     result += text + " "
    # result = result.strip()
    result = " ".join(inputs.texts)
    return OutputText(result=result)

# odpalenie pliku
# python -m fastapi dev model_serving_1.py
