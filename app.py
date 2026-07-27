"""
Small Flask API serving the placement classifier.

POST /predict with a student's profile -> returns placement prediction
plus the probability, so it's honest about confidence, not just a label.

Run locally:   python app.py
Then:          curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @sample_request.json
"""
from pathlib import Path

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

MODEL = joblib.load(BASE_DIR / "models" / "placement_classifier.pkl")
ENCODERS = joblib.load(BASE_DIR / "models" / "encoders.pkl")
SALARY_REGRESSOR = None

FEATURE_ORDER = [
    "gender", "ssc_p", "ssc_b", "hsc_p", "hsc_b", "hsc_s",
    "degree_p", "degree_t", "workex", "etest_p", "specialisation", "mba_p",
]

SCORE_FIELDS = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]


def load_salary_regressor():
    global SALARY_REGRESSOR
    if SALARY_REGRESSOR is None:
        SALARY_REGRESSOR = joblib.load(BASE_DIR / "models" / "salary_regressor.pkl")
    return SALARY_REGRESSOR


def validate_score_fields(payload):
    for field in SCORE_FIELDS:
        value = payload[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return jsonify({"error": f"'{field}' must be a number between 0 and 100"}), 400
        if value < 0 or value > 100:
            return jsonify({"error": f"'{field}' must be between 0 and 100"}), 400
    return None


def encode_categorical_value(field, value):
    if field not in ENCODERS:
        return value, None

    try:
        encoded = ENCODERS[field].transform([value])[0]
    except (ValueError, TypeError):
        valid_options = list(ENCODERS[field].classes_)
        return None, (jsonify({
            "error": f"invalid value '{value}' for '{field}'",
            "valid_options": valid_options,
        }), 400)

    return encoded, None


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "placement-predictor API is running"})


@app.route("/app", methods=["GET"])
def serve_app():
    from flask import send_from_directory
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    payload = request.get_json(force=True)

    missing = [f for f in FEATURE_ORDER if f not in payload]
    if missing:
        return jsonify({"error": f"missing fields: {', '.join(missing)}"}), 400

    validation_error = validate_score_fields(payload)
    if validation_error is not None:
        return validation_error

    row = {}
    for col in FEATURE_ORDER:
        val = payload[col]
        if col in ENCODERS and col != "status":
            val, error_response = encode_categorical_value(col, val)
            if error_response is not None:
                return error_response
        row[col] = val

    X = pd.DataFrame([row])[FEATURE_ORDER]
    pred = MODEL.predict(X)[0]
    proba = MODEL.predict_proba(X)[0]

    status_label = ENCODERS["status"].inverse_transform([pred])[0]
    placed_idx = list(ENCODERS["status"].classes_).index("Placed")

    predicted_salary = None
    if status_label == "Placed":
        salary_model = load_salary_regressor()
        predicted_salary = round(float(salary_model.predict(X)[0]), 2)

    return jsonify({
        "prediction": status_label,
        "placement_probability": round(float(proba[placed_idx]), 3),
        "predicted_salary": predicted_salary,
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    try:
        app.run(host="0.0.0.0", port=port, debug=True)
    except OSError:
        alt_port = 5001
        print(f"Port {port} is in use (e.g. AirPlay Receiver on macOS). Launching on port {alt_port}...")
        app.run(host="0.0.0.0", port=alt_port, debug=True)
