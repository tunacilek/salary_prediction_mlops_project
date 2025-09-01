from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field

class Salary(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    Age: float
    Gender: str
    education_level: str
    job_title: str
    years_of_experience: float
    prediction: float
    prediction_time: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class RequestSalary(SQLModel):
    Age: float
    Gender: str
    education_level: str
    job_title: str
    years_of_experience: float

class SalaryTrain(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    Age: float
    Gender: str
    education_level: str
    job_title: str
    years_of_experience: float
    Salary: float

class SalaryDriftInput(SQLModel):
    n_days_before: int

    class Config:
        json_schema_extra = {
            "example": {
                "n_days_before": 5,
            }
        }