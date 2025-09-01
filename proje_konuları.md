# VBO MLOps Bootcamp Final Project Assignment

## Overview
For the final project, you will design, develop, and deploy a machine learning **(ML) solution of your choice**. You are free to select your own dataset and problem statement (e.g., classification, regression, time-series forecasting, etc.), provided it is suitable for an ML approach. The **focus of this project is to apply MLOps principles**, including model development, pipeline creation, deployment, and infrastructure management, using one of the specified infrastructure combinations.

---

## Project Requirements

### 1. Problem Definition and Dataset
- Choose a dataset and define an ML problem of your interest (e.g., predicting house prices, classifying images, forecasting sales, etc.).
- Provide a brief description of the dataset and problem in your project documentation. Keep it simple, no more than a few pages/slides.

### 2. Model Development
- **2.1.** Develop an ML model to solve your chosen problem.
- **2.2.** Use an ML pipeline (if possible) to preprocess data, train the model, and evaluate its performance.
- **2.3.** Document the modelâ€™s performance metrics (e.g., accuracy, RMSE, F1-score) and justify your choice of model. Maybe MLFlow would be good idea :)

### 3. Deployment
- **3.1.** Create an API (FastAPI would be great) to serve your modelâ€™s predictions with at least one endpoint.
- **3.2.** Implement a mechanism to **detect model concept drift or data drift** (e.g., statistical tests, monitoring predictions over time).
- **3.5.** Automate the deployment process using a CI/CD tool such as Jenkins, Ansible or Gitea.

### 4. Infrastructure Options
You must implement your project using **one** of the following infrastructure combinations. Each option has specific requirements:

#### a. With Database, On Docker
- Use Docker containers for the application.
- Store prediction results in a MySQL database.
- Ensure the API writes predictions to the database.

#### b. Without Database, On Minikube
- Deploy the application on Minikube (local Kubernetes).
- Serve predictions without persistent storage (no database required).

#### c. Without Database, On AWS EC2 Instance
- Deploy the application on an AWS EC2 instance.
- Serve predictions without persistent storage.

#### d. With Database, With UI
- Use Docker containers for the application.
- Store prediction results in a MySQL database.
- Develop a simple UI (e.g., using Flask, Streamlit, or similar) to interact with the API.

#### e. Without Database, On AWS Elastic Kubernetes Service (EKS)
- Deploy the application on AWS EKS.
- Serve predictions without persistent storage.

#### f. Without Database, On AWS EC2 Instance, Using Terraform and Ansible
- Deploy the application on an AWS EC2 instance.
- Use Terraform to provision the infrastructure and Ansible to configure and deploy the application.
- Serve predictions without persistent storage.

#### g. On AWS EC2 Instance, Package and Push to PyPI, Use EC2 User Script
- Deploy the application on an AWS EC2 instance.
- Package your code as a Python library and push it to PyPI.
- Provide a script for installation and execution (e.g., via `pip install` and a machine user script).
- Serve predictions without persistent storage.

---

## Deliverables
1. **Demo**: A working API endpoint (deployed or locally demonstrable) with example requests and responses.
2. **Codebase**: A well-structured repository (e.g., on GitHub/GitLab) containing:
   - ML model code and pipeline.
   - API implementation.
   - Infrastructure setup files (e.g., Dockerfile, Terraform scripts, Ansible playbooks, etc., depending on your chosen option).
   - Deployment automation scripts (e.g., Jenkins pipeline).
3. **Documentation**: A README or report including:
   - Problem statement and dataset description.
   - Model details (approach, performance metrics).
   - Instructions to set up and run the project locally and on the chosen infrastructure.
   - Explanation of the drift detection mechanism.


---

## Evaluation Criteria
- **Correctness**: Does the ML model solve the chosen problem effectively?
- **MLOps Practices**: Are pipelines, automation, and drift detection implemented properly?
- **Infrastructure**: Is the chosen infrastructure combination correctly set up and functional?
- **Code Quality**: Is the code modular, documented, and maintainable?
- **Creativity**: Is the problem/dataset choice interesting and well-justified?

---

## Notes
- If you encounter issues with a specific infrastructure (e.g., AWS costs), consult the mentor/instructor for alternatives or simplifications.
- You will have only 10 minutes to perform online demo.
- You can set up a team with your friend. In this case evaluators will measure equal contribution of the team members.

Good luck, and have fun building your MLOps solution!