# Heart Disease Prediction Web App

A **Flask-based web application** that predicts the **10-year risk of heart disease** using a **Random Forest Classifier**. Users can enter patient data through a web form, and the app outputs a **prediction, probability, and risk explanation**.

---

## Features

- **Random Forest model** trained on relevant cardiovascular data.  
- Input form for **14 patient features** (age, gender, smoking, blood pressure, cholesterol, BMI, etc.).  
- Predicts **Low, Moderate, High, or Very High risk**.  
- Displays **probability** and **friendly explanation** for each risk category.  
- Home page explains each feature and why the **education column was dropped**.  
- Simple and clean user interface with a **Back to Home** button.  
- Ready to deploy on **Hugging Face Spaces or locally**.

---

## Dataset and Model

- **Target variable**: `TenYearCHD` (10-year risk of coronary heart disease).  
- **Features used** (education dropped):  

| Feature | Description |
|---------|-------------|
| male | Sex: 1 = male, 0 = female |
| age | Age in years |
| currentSmoker | Currently smoking? 1 = yes, 0 = no |
| cigsPerDay | Average number of cigarettes per day |
| BPMeds | On blood pressure medication? 1 = yes, 0 = no |
| prevalentStroke | History of stroke? 1 = yes, 0 = no |
| prevalentHyp | History of hypertension? 1 = yes, 0 = no |
| diabetes | Diabetes? 1 = yes, 0 = no |
| totChol | Total cholesterol in mg/dL |
| sysBP | Systolic blood pressure (mm Hg) |
| diaBP | Diastolic blood pressure (mm Hg) |
| BMI | Body Mass Index (kg/m²) |
| heartRate | Heart rate in beats per minute |
| glucose | Blood glucose level in mg/dL |

- **Note**: The `education` column was dropped because it added little predictive power.  
- **Model**: Random Forest Classifier saved as `heartDiseaseprdcition.pkl` using `joblib`.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/jashandeep032kaur-dot/heart-disease-prediction-model-2.git
cd heart-disease-prediction-model-2

---
title: Heart Disease Prediction Model2
emoji: 🐨
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
