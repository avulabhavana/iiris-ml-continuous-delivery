
from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "iris_model.pkl"


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Iris Prediction API is running",
        "endpoint": "/predict"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({
            "error": "Model file not found"
        }), 500

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a valid JSON object"
        }), 400

    features = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]

    if not all(feature in data for feature in features):
        return jsonify({
            "error": "Provide all four Iris measurements"
        }), 400

    try:
        values = [float(data[feature]) for feature in features]

        if not all(pd.notna(value) for value in values):
            raise ValueError

        input_data = pd.DataFrame([values], columns=features)
        model = joblib.load(MODEL_PATH)
        prediction = model.predict(input_data)[0]

        return jsonify({
            "predicted_species": str(prediction)
        })

    except (ValueError, TypeError):
        return jsonify({
            "error": "Measurements must be valid numbers"
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
