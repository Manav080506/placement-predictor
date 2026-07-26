"""
Small Flask API serving the placement classifier.

POST /predict with a student's profile -> returns placement prediction
plus the probability, so it's honest about confidence, not just a label.

Run locally:   python app.py
Then:          curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @sample_request.json
"""
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

MODEL = joblib.load("models/placement_classifier.pkl")
ENCODERS = joblib.load("models/encoders.pkl")

FEATURE_ORDER = [
    "gender", "ssc_p", "ssc_b", "hsc_p", "hsc_b", "hsc_s",
    "degree_p", "degree_t", "workex", "etest_p", "specialisation", "mba_p",
]


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "placement-predictor API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)

    missing = [f for f in FEATURE_ORDER if f not in payload]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400

    row = {}
    for col in FEATURE_ORDER:
        val = payload[col]
        if col in ENCODERS:
            try:
                val = ENCODERS[col].transform([val])[0]
            except ValueError:
                return jsonify({
                    "error": f"invalid value '{val}' for '{col}'. "
                             f"Expected one of {list(ENCODERS[col].classes_)}"
                }), 400
        row[col] = val

    X = pd.DataFrame([row])[FEATURE_ORDER]
    pred = MODEL.predict(X)[0]
    proba = MODEL.predict_proba(X)[0]

    status_label = ENCODERS["status"].inverse_transform([pred])[0]
    placed_idx = list(ENCODERS["status"].classes_).index("Placed")

    return jsonify({
        "prediction": status_label,
        "placement_probability": round(float(proba[placed_idx]), 3),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
