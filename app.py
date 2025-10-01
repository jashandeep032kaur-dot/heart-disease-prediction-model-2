from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# ------------------------------
# Load trained model
# ------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "heartDiseasePredction2.pkl")
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print("⚠️ Error loading model:", e)
else:
    print("⚠️ Model file not found. Please upload model.pkl")

# ------------------------------
# Features (education dropped)
# ------------------------------
FEATURES = [
    "male", "age", "currentSmoker", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
    "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]

# ------------------------------
# Home Page
# ------------------------------
@app.route("/")
def index():
    column_info = {
        "male": "Sex: 1 = male, 0 = female",
        "age": "Age in years",
        "currentSmoker": "Currently smoking? 1 = yes, 0 = no",
        "cigsPerDay": "Average cigarettes per day",
        "BPMeds": "On BP meds? 1 = yes, 0 = no",
        "prevalentStroke": "History of stroke? 1 = yes, 0 = no",
        "prevalentHyp": "History of hypertension? 1 = yes, 0 = no",
        "diabetes": "Diabetes? 1 = yes, 0 = no",
        "totChol": "Total cholesterol (mg/dL)",
        "sysBP": "Systolic BP (mm Hg)",
        "diaBP": "Diastolic BP (mm Hg)",
        "BMI": "Body Mass Index (kg/m²)",
        "heartRate": "Heart rate (bpm)",
        "glucose": "Glucose (mg/dL)"
    }
    return render_template(
        "index.html",
        features=FEATURES,
        column_info=column_info,
        education_explain="The 'education' column was dropped as it added little predictive power."
    )

# ------------------------------
# Prediction Form Page
# ------------------------------
@app.route("/predict")
def predict_form():
    form_meta = [
        {"name": "male", "label": "Sex (1=male,0=female)", "type": "number", "min": "0", "max": "1"},
        {"name": "age", "label": "Age", "type": "number", "min": "0"},
        {"name": "currentSmoker", "label": "Current Smoker (1=yes,0=no)", "type": "number", "min": "0", "max": "1"},
        {"name": "cigsPerDay", "label": "Cigarettes per Day", "type": "number", "min": "0"},
        {"name": "BPMeds", "label": "BP Meds (1=yes,0=no)", "type": "number", "min": "0", "max": "1"},
        {"name": "prevalentStroke", "label": "Stroke History (1=yes,0=no)", "type": "number", "min": "0", "max": "1"},
        {"name": "prevalentHyp", "label": "Hypertension (1=yes,0=no)", "type": "number", "min": "0", "max": "1"},
        {"name": "diabetes", "label": "Diabetes (1=yes,0=no)", "type": "number", "min": "0", "max": "1"},
        {"name": "totChol", "label": "Total Cholesterol (mg/dL)", "type": "number", "min": "0"},
        {"name": "sysBP", "label": "Systolic BP (mm Hg)", "type": "number", "min": "0"},
        {"name": "diaBP", "label": "Diastolic BP (mm Hg)", "type": "number", "min": "0"},
        {"name": "BMI", "label": "Body Mass Index (kg/m²)", "type": "number", "step": "0.1"},
        {"name": "heartRate", "label": "Heart Rate (bpm)", "type": "number", "min": "0"},
        {"name": "glucose", "label": "Glucose (mg/dL)", "type": "number", "min": "0"},
    ]
    return render_template("predict.html", form_meta=form_meta)

# ------------------------------
# Prediction Result Page
# ------------------------------
@app.route("/result", methods=["POST"])
def result():
    if model is None:
        return render_template("result.html", error="⚠️ Model not loaded. Please upload model.pkl")

    try:
        # Collect input values
        values = [float(request.form[feat]) for feat in FEATURES]
        X = np.array(values).reshape(1, -1)

        # Prediction
        pred = model.predict(X)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0][1]

        # Result interpretation
        label = "High Risk" if pred == 1 else "Low Risk"
        prob_text = f"{proba*100:.2f}%" if proba is not None else "N/A"

        return render_template("result.html",
                               prediction=pred,
                               label=label,
                               probability=prob_text)

    except Exception as e:
        return render_template("result.html", error=f"⚠️ Error during prediction: {e}")

# ------------------------------
# Run locally (for testing)
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
