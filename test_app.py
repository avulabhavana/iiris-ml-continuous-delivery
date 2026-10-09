
import joblib
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import app as iris_app


FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    iris = load_iris(as_frame=True)

    X = iris.data.copy()
    X.columns = FEATURES
    y = iris.target_names[iris.target]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X, y)

    model_path = tmp_path / "iris_model.pkl"
    joblib.dump(model, model_path)

    monkeypatch.setattr(
        iris_app,
        "MODEL_PATH",
        str(model_path)
    )

    iris_app.app.config["TESTING"] = True

    with iris_app.app.test_client() as test_client:
        yield test_client


def test_home_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json["message"] == (
        "Iris Prediction API is running"
    )


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

    # Print the actual API response if the test fails.
    print("Status code:", response.status_code)
    print("Response body:", response.get_data(as_text=True))

    assert response.status_code == 200, (
        f"API returned {response.status_code}: "
        f"{response.get_data(as_text=True)}"
    )

    assert response.json["predicted_species"] == "setosa"


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
