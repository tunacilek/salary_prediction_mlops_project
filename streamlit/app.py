import streamlit as st
import requests
import math

FASTAPI_URL = "http://host.docker.internal:8000/prediction/salary"

st.title("Salary Prediction App")
st.write("Enter Your Data Below:")

Age = st.number_input("Age:", min_value=18, max_value=100)
years_of_experience = st.number_input("Years of Experience:", min_value=0, max_value=60)
education_level = st.selectbox("Education Level:", ["Bachelor's Degree", "Master's Degree", "PhD", "High School"])
job_title = st.selectbox("Job Title::", ["Software Engineer", "Data Scientist", "Software Engineer Manager", "Data Analyst", "Senior Project Engineer", "Product Manager", "Full Stack Engineer", "Marketing Manager", "Senior Software Engineer", "Back end Developer", "Front end Developer", "Software Developer", "Marketing Coordinator", "Junior Sales Associate", "Financial Manager", "Marketing Analyst", "Operations Manager", "Human Resources Manager"])
Gender = st.selectbox("Gender:", ["Male", "Female"])

if st.button("Predict"):

    # Preapare the input data
    input_data = {
            "Age": Age,
            "years_of_experience": years_of_experience,
            "education_level": education_level,
            "job_title": job_title,
            "Gender": Gender
        }

    try:
        response = requests.post(FASTAPI_URL, json=input_data)
        
        if response.status_code == 200:
            prediction = response.json()
            print(prediction)
            predicted_salary = math.ceil(prediction["predicted_salary"])
            st.success(f"Predicted Salary: **${predicted_salary}**")
        else:
            st.error(f"Error: Received status code {response.status_code}")

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")