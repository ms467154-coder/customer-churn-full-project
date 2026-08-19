from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def profile():
    return {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_summary_contains_real_dataset_metrics():
    response = client.get("/api/summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["customers"] == 7043
    assert 0 < payload["churn_rate"] < 1


def test_prediction_returns_risk_assessment():
    response = client.post("/api/predict", json=profile())
    assert response.status_code == 200
    payload = response.json()
    assert 0 <= payload["churn_probability"] <= 1
    assert payload["risk_level"] in {"Low", "Medium", "High"}
    assert payload["recommendation"]


def test_prediction_validates_required_fields():
    response = client.post("/api/predict", json={})
    assert response.status_code == 422
