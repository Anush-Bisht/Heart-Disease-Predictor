# Heart Disease Predictor

A web app that estimates heart disease risk from clinical data.
Built with scikit-learn (KNN), Flask, HTML, CSS and JavaScript.

## Run locally
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python train_model.py
    python app.py
Then open http://127.0.0.1:5000

## Model
KNN pipeline (encoding + scaling) with about 86% cross-validated accuracy.
Dataset: Heart Failure Prediction (Kaggle), 918 patients.

## Disclaimer
For education only. Not a medical diagnosis.