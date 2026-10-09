
from flask import Flask, request, jsonify
import joblib
import math
import os
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "iris_model.pkl"

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Iris Prediction API is running",
        "endpoint": "/predict"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    # Check whether the trained model exists
    if not os.path.exists(MODEL_PATH):
        return jsonify({
            "error": "Model file not found"
        }), 500

    # Read JSON input
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a valid JSON object"
        }), 400

    # Check that all four measurements are provided
    if not all(feature in data for feature in FEATURES):
        return jsonify({
            "error": "Provide all four Iris measurements"
        }), 400

    # Validate the measurements
    try:
        values = [float(data[feature]) for feature in FEATURES]

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Measurements must be finite numbers")

        if any(value <= 0 for value in values):
            raise ValueError("Measurements must be positive")

    except (ValueError, TypeError, OverflowError):
        return jsonify({
            "error": "Measurements must be valid positive numbers"
        }), 400

    # Prepare the input using the expected feature names
    input_data = pd.DataFrame([values], columns=FEATURES)

    try:
        model = joblib.load(MODEL_PATH)
        prediction = model.predict(input_data)[0]

        return jsonify({
            "predicted_species": str(prediction)
        }), 200

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed",
            "details": str(error)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
