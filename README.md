# Explainable ML API for Risk Classification

## Overview

This project implements an **Explainable Machine Learning API** for loan risk classification.

The API predicts whether a loan is likely to default and provides **model explanations using SHAP values**. The service supports **single predictions, batch predictions, and explainability endpoints**, including **SHAP waterfall visualizations**.

The system is built using **FastAPI**, **scikit-learn**, **SHAP**, and **Docker**, making it suitable for production-style machine learning deployment.

---

## Features

- Loan risk prediction using a trained **Random Forest classifier**
- Model explainability using **SHAP values**
- **SHAP waterfall plot** generation for individual predictions
- **Batch prediction** using CSV uploads
- **Batch explanation** of predictions
- **Dockerized deployment** for reproducibility
- Interactive API documentation using **Swagger UI**

---

## Tech Stack

- **Python**
- **FastAPI**
- **scikit-learn**
- **SHAP**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Docker**

---

## Project Structure

```text
<<<<<<< HEAD
explainable-ml-api/
│
├── main.py                 # FastAPI application
├── Dockerfile              # Docker container configuration
├── requirements.txt        # Python dependencies
├── risk_model.pkl          # Trained Random Forest model
├── feature_names.pkl       # Feature names used by the model
├── model_train_test.py     # Script used to train the model
├── test.http               # API test requests
├── README.md               # Project documentation
├── .dockerignore           # Files ignored during Docker build
```

---

## API Endpoints

| Endpoint | Method | Description |
|--------|--------|-------------|
| `/` | GET | API welcome message |
| `/health` | GET | Health check |
| `/features` | GET | Returns model feature names |
| `/predict` | POST | Predict loan risk |
| `/explain` | POST | Explain prediction using SHAP |
| `/batch_predict` | POST | Batch prediction using CSV |
| `/batch_explain` | POST | Batch explanation using CSV |
| `/explain_waterfall_png` | POST | Generate SHAP waterfall visualization |

---

## Example Prediction Request

### Endpoint


POST /predict


### Request Body

```json
{
  "person_age": 22,
  "person_income": 59000,
  "person_emp_length": 3,
  "loan_amnt": 10000,
  "loan_int_rate": 12.5,
  "loan_percent_income": 0.2,
  "cb_person_cred_hist_length": 3,
  "person_home_ownership": "RENT",
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "cb_person_default_on_file": "N"
}
```


### Response

```json
{
  "prediction": 0,
  "risk_score": 0.12
}
```

## Running the Project Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI Server

```bash
uvicorn main:app --reload
```

### 3. Open API Documentation

http://127.0.0.1:8000/docs

FastAPI automatically provides **interactive Swagger documentation** where all endpoints can be tested.


---

## Running with Docker

### Build the Docker Image

```bash
docker build -t explainable-ml-api .
```

---

### Run the Container

```bash
docker run -p 8000:8000 explainable-ml-api
```

---

### Access the API


http://127.0.0.1:8000/docs


---

## Example Batch Prediction

Upload a **CSV file** containing the same columns as the API input fields.

### Example CSV Format

```csv
person_age,person_income,person_emp_length,loan_amnt,loan_int_rate,loan_percent_income,cb_person_cred_hist_length,person_home_ownership,loan_intent,loan_grade,cb_person_default_on_file
22,59000,3,10000,12.5,0.2,3,RENT,PERSONAL,B,N
30,85000,6,15000,10.5,0.18,5,MORTGAGE,EDUCATION,A,N
```

Use endpoint:

```
POST /batch_predict
```

or

```
POST /batch_explain
```

---

## Model Explainability

The API uses **SHAP (SHapley Additive exPlanations)** to explain model predictions.

For each prediction, the API returns:

- Base value (average model prediction)
- Feature contributions
- Direction of impact on risk
- SHAP waterfall visualization

This allows users to understand **why a prediction was made**.

---

## Author

**Poornima Jayadevan**  
Master’s Student in Artificial Intelligence  
Brandenburg University of Technology Cottbus-Senftenberg
