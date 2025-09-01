from pydantic import BaseModel

class SalaryInput(BaseModel):
    Age: float
    years_of_experience: float
    Gender: str
    education_level: str
    job_title: str

class PredictionOutput(BaseModel):
    predicted_salary: float