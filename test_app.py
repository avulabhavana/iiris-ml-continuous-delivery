
import joblib
import pytest
from sklearn.datasets import load_iris

import app as iris_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    model_path = tmp_path / "iris_model.pkl"

    iris_data = load_iris(as_frame=True)
    model = joblib.load("iris_model.pkl") if False else None

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    X = iris_data.data
    y = iris_data.target

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X, y)

    joblib.dump(pipeline, model_path)
    monkeypatch.setattr(iris_app, "MODEL_PATH", str(model_path))

    iris_app.app.config["TESTING"] = True

    with iris_app.app.test_client() as test_client:
        yield test_client


def test_home_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json["message"] == "Iris Prediction API is running"


def test_valid_prediction(client):
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 200
    assert response.json["predicted_species"] in [
        "setosa",
        "versicolor",
        "virginica",
        "0",
        "1",
        "2",
    ]


def test_missing_features(client):
    response = client.post(
        "/predict",
        json={"sepal_length": 5.1},
    )

    assert response.status_code == 400


def test_invalid_measurements(client):
    response = client.post(
        "/predict",
        json={
            "sepal_length": "abc",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 400
