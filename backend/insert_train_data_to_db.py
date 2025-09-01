import pandas as pd
from sqlalchemy.sql import text as sa_text
from database import engine
from models import SalaryTrain
from sqlmodel import Session, SQLModel
from sklearn.model_selection import train_test_split

SQLModel.metadata.create_all(engine)

df = pd.read_csv('Salary_Data.csv')

X = df.drop(columns=['Salary'])
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

df_train = pd.concat([X_train, y_train], axis=1)

print(df_train.head())

with Session(engine) as session:
    records_to_insert = []

    for _, row in df_train.iterrows():
        record = SalaryTrain(
            Age=row["Age"],
            Gender=row["Gender"],
            education_level=row["education_level"],
            job_title=row["job_title"],
            years_of_experience=row["years_of_experience"],
            Salary=row["Salary"]
        )
        records_to_insert.append(record)

    session.bulk_save_objects(records_to_insert)
    session.commit()