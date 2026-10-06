import joblib, pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
bundle = joblib.load("model.pkl")
pipe, FEATURES, MEANS = bundle["pipeline"], bundle["features"], bundle["means"]

CHOICES = {
    "Sex": {"M", "F"},
    "ChestPainType": {"ASY", "ATA", "NAP", "TA"},
    "RestingECG": {"Normal", "ST", "LVH"},
    "ExerciseAngina": {"Y", "N"},
    "ST_Slope": {"Up", "Flat", "Down"},
}
RANGES = {  # min, max
    "Age": (1, 120), "RestingBP": (50, 250), "Cholesterol": (50, 700),
    "MaxHR": (50, 230), "Oldpeak": (-3, 7),
}

@app.route("/")
def home():
    return render_template("index.html", accuracy=bundle["accuracy"])

@app.post("/predict")
def predict():
    d = request.get_json(silent=True) or {}
    row = {}
    try:
        for k, (lo, hi) in RANGES.items():
            raw = d.get(k)
            if k in ("Cholesterol", "RestingBP") and raw in (None, "", 0, "0"):
                v = MEANS[k]                      # blank -> training mean
            else:
                v = float(raw)
            if not lo <= v <= hi:
                return jsonify(error=f"{k} must be between {lo} and {hi}."), 400
            row[k] = v
        row["FastingBS"] = int(d.get("FastingBS"))
        if row["FastingBS"] not in (0, 1):
            raise ValueError
        for k, allowed in CHOICES.items():
            if d.get(k) not in allowed:
                return jsonify(error=f"Invalid value for {k}."), 400
            row[k] = d[k]
    except (TypeError, ValueError):
        return jsonify(error="Please fill in every field with valid numbers."), 400

    X = pd.DataFrame([row])[FEATURES]
    prob = float(pipe.predict_proba(X)[0][1])
    return jsonify(probability=round(prob * 100, 1), prediction=int(prob >= 0.5))

if __name__ == "__main__":
    app.run(debug=True)
