from fastapi import FastAPI
from pydantic import BaseModel,Field
import pickle

app = FastAPI()
class StudentData(BaseModel):
    study_hour: float = Field(ge=0)
    attendance: float = Field(ge=0,le=100)
    assignment_completed: int = Field(ge=0,le=1)
    previous_score: float = Field(ge=0,le=100)

with open("Student_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.get("/")
def home():
    return {"Message " : "ML API is Running"}

@app.post("/predict")
def predict(data: StudentData):
    prediction = model.predict([[
        data.study_hour,
        data.attendance,
        data.assignment_completed,
        data.previous_score
    ]])

    return {"prediction": int(prediction[0])}

@app.get("/predict-test")
def predict_test(
    study_hour: float,
    attendance: float,
    assignment_completed: int,
    previous_score: float
):
    prediction = model.predict([[
        study_hour,
        attendance,
        assignment_completed,
        previous_score
    ]])

    result = "Pass" if prediction[0] == 1 else "Fail"
    return {"prediction": int(prediction[0]), "result": result}

@app.get("/predict-test")
def predict_test(
    study_hour: float,
    attendance: float,
    assignment_completed: int,
    previous_score: float
):
    prediction = model.predict([[
        study_hour,
        attendance,
        assignment_completed,
        previous_score
    ]])

    return {"prediction": int(prediction[0])}