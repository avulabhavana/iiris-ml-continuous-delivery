
import json
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = "iris.csv"
MODEL_PATH = "iris_model.pkl"
METRICS_PATH = "metrics.json"

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET = "species"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found in the repository"
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing = set(required_columns) - set(data.columns)

    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "accuracy": float(accuracy),
        "model": "LogisticRegression",
        "dataset": DATA_PATH,
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
