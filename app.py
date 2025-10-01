from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Path to the trained model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Upload model.pkl")

# Load model with joblib
model = joblib.load(MODEL_PATH)

# Features used in training (education dropped)
FEATURES = [
    "male", "age", "currentSmoker", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
    "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]

supports_proba = hasattr(model, "predict_proba")


@app.route("/")
def index():
    column_info = {
        "male": "Sex: 1 = male, 0 = female",
        "age": "Age in years",
        "currentSmoker": "Currently smoking? 1 = yes, 0 = no",
        "cigsPerDay": "Average number of cigarettes smoked per day",
        "BPMeds": "On blood pressure medication? 1 = yes, 0 = no",
        "prevalentStroke": "History of stroke? 1 = yes, 0 = no",
        "prevalentHyp": "History of hypertension? 1 = yes, 0 = no",
        "diabetes": "Diabetes diagnosis? 1 = yes, 0 = no",
        "totChol": "Total cholesterol (mg/dL)",
        "sysBP": "Systolic blood pressure (mm Hg)",
        "diaBP": "Diastolic blood pressure (mm Hg)",
        "BMI": "Body Mass Index (kg/m²)",
        "heartRate": "Heart rate (beats per minute)",
        "glucose": "Glucose level (mg/dL)"
    }
    education_explain = (
        "The 'education' column was dropped because it added little predictive power "
        "and could introduce socioeconomic bias. The model performs well without it."
    )
    return render_template("index.html",
                           features=FEATURES,
                           column_info=column_info,
                           education_explain=education_explain)


@app.route("/predict", methods=["GET"])
def predict_form():
    form_meta = [
        {"name": "male", "label": "Sex (1=male,0=female)", "type": "number", "min": "0", "max": "1"},
        {"name": "age", "label": "Age (years)", "type": "number", "min": "0"},
        {"name": "currentSmoker", "label": "Current Smoker (1=yes,0=no)", "type": "number", "min": "0", "max": "1"},
        {"name": "cigsPerDay", "label": "Cigarettes per Day", "type": "number", "min": "0"},
        {"name": "BPMeds", "label": "On BP Meds (1=yes,0=no)", "type": "number", "min": "0", "max": "1"},
        {"name": "prevalentStroke", "label": "History of Stroke (1=yes,0=no)", "type": "number", "min": "0", "max": "1"},
        {"name": "prevalentHyp", "label": "History of Hypertension (1=yes,0=no)", "type": "number", "min": "0", "max": "1"},
        {"name": "diabetes", "label": "Diabetes (1=yes,0=no)", "type": "number", "min": "0", "max": "1"},
        {"name": "totChol", "label": "Total Cholesterol (mg/dL)", "type": "number", "min": "0"},
        {"name": "sysBP", "label": "Systolic BP (mm Hg)", "type": "number", "min": "0"},
        {"name": "diaBP", "label": "Diastolic BP (mm Hg)", "type": "number", "min": "0"},
        {"name": "BMI", "label": "Body Mass Index", "type": "number", "step": "0.1", "min": "0"},
        {"name": "heartRate", "label": "Heart Rate (bpm)", "type": "number", "min": "0"},
        {"name": "glucose", "label": "Glucose (mg/dL)", "type": "number", "min": "0"},
    ]
    return render_template("predict.html", form_meta=form_meta)


@app.route("/result", methods=["POST"])
def result():
    try:
        values = [float(request.form[feat]) for feat in FEATURES]
        X = np.array(values).reshape(1, -1)

        if supports_proba:
            proba = model.predict_proba(X)[0][1]
            pred = int(model.predict(X)[0])
        else:
            pred = int(model.predict(X)[0])
            proba = None

    except Exception as e:
        return render_template("result.html", error=f"Error: {str(e)}")

    label = "High Risk (Positive)" if pred == 1 else "Low Risk (Negative)"
    prob_text = f"{proba*100:.2f}%" if proba is not None else "N/A"
    return render_template("result.html", prediction=pred, label=label, probability=prob_text)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
