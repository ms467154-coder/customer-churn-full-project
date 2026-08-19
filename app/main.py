from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
STATIC_PATH = ROOT / "frontend"

app = FastAPI(title="Telco Churn Intelligence API", version="1.0.0", description="Production-style inference API for customer churn risk.")
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")

TARGET = "Churn"
DROP_COLUMNS = ["customerID", TARGET]

class CustomerInput(BaseModel):
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, le=100)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


def load_model() -> tuple[Pipeline, list[str], float]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    X = df.drop(columns=DROP_COLUMNS)
    y = (df[TARGET] == "Yes").astype(int)
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    model.fit(X, y)
    return model, X.columns.tolist(), float(y.mean())

model, FEATURE_COLUMNS, BASELINE_CHURN_RATE = load_model()

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_PATH / "index.html")

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "model": "logistic-regression", "dataset": "telco-customer-churn"}

@app.get("/api/summary")
def summary() -> dict[str, Any]:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    churned = df[df[TARGET] == "Yes"]
    return {
        "customers": int(len(df)),
        "churn_rate": round(float((df[TARGET] == "Yes").mean()), 4),
        "average_monthly_charges": round(float(df["MonthlyCharges"].mean()), 2),
        "average_tenure_months": round(float(df["tenure"].mean()), 1),
        "top_risk_segments": [
            {"label": "Month-to-month contracts", "rate": round(float((churned["Contract"] == "Month-to-month").sum() / max((df["Contract"] == "Month-to-month").sum(), 1)), 3)},
            {"label": "Electronic check payments", "rate": round(float((churned["PaymentMethod"] == "Electronic check").sum() / max((df["PaymentMethod"] == "Electronic check").sum(), 1)), 3)},
            {"label": "Fiber optic customers", "rate": round(float((churned["InternetService"] == "Fiber optic").sum() / max((df["InternetService"] == "Fiber optic").sum(), 1)), 3)},
        ],
    }

@app.post("/api/predict")
def predict(customer: CustomerInput) -> dict[str, Any]:
    try:
        payload = pd.DataFrame([customer.model_dump()])[FEATURE_COLUMNS]
        probability = float(model.predict_proba(payload)[0, 1])
        risk = "High" if probability >= 0.65 else "Medium" if probability >= 0.35 else "Low"
        recommendation = {"High": "Prioritize a retention call and review contract or pricing options.", "Medium": "Offer proactive support and a plan review before the next billing cycle.", "Low": "Maintain service quality and monitor for changes in usage or tenure."}[risk]
        return {"churn_probability": round(probability, 4), "risk_level": risk, "recommendation": recommendation, "baseline_churn_rate": BASELINE_CHURN_RATE}
    except Exception as exc:
        raise HTTPException(status_code=422, detail="The customer profile could not be processed. Check the submitted values.") from exc
