# Churn Intelligence

A production-style customer churn risk application built around the existing Telco Customer Churn analysis by **Mohamed Salem**. The original notebook and dataset remain intact; the application adds a clean inference API and responsive user interface around the real data and model workflow.

> Identify high-risk customers early so retention teams can focus their time where it matters most.

## What was built

The project now includes a FastAPI application that trains a reproducible scikit-learn pipeline from the supplied dataset at startup, validates customer profiles with typed request schemas, returns calibrated risk probabilities and business-friendly recommendations, and exposes health and real-data summary endpoints. A responsive cream/off-white and muted-red interface provides the main prediction workflow, portfolio metrics, loading states, validation feedback, and mobile-friendly layouts.

| Area | Implementation |
|---|---|
| Existing analysis | `customer_churn.ipynb` retained without removing the original exploration |
| Data | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| ML | Numeric imputation/scaling, categorical imputation/one-hot encoding, class-weighted logistic regression |
| Backend | FastAPI REST API with `/api/health`, `/api/summary`, and `/api/predict` |
| Frontend | Accessible semantic HTML/CSS/JavaScript served by FastAPI |
| Testing | Pytest coverage for health, summary, prediction, and validation behavior |
| Branding | Minimal cream/off-white system with muted red accents and subtle Mohamed Salem attribution |

## Quick start

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Interactive API documentation is available at `/docs`.

## API examples

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Prediction requests require the customer profile fields represented in the web form. The response contains `churn_probability`, `risk_level`, `recommendation`, and the observed dataset baseline churn rate.

## Testing

Run the automated tests from the project root:

```bash
pytest -q
```

The application uses the supplied dataset as its source of truth and does not generate fake business metrics. The model is retrained in memory at application startup, which keeps the repository portable and avoids committing generated model binaries. For a larger deployment, the next production step would be to version a serialized model artifact through an approved model registry and add monitoring for feature drift and prediction quality.

## Architecture

```text
Browser UI (frontend/index.html)
          |
          v
FastAPI application (app/main.py)
          |
          v
Validated CustomerInput -> preprocessing pipeline -> logistic regression
          |
          v
Telco Customer Churn CSV
```

## Important limitations

This is a decision-support prototype rather than an autonomous retention system. It does not provide authentication, persistent prediction history, model registry integration, or production monitoring because those capabilities were not present in the original project and would require business and infrastructure decisions. The existing notebook is intentionally preserved as the original analysis artifact.

## Project structure

```text
.
├── app/main.py
├── frontend/index.html
├── tests/test_api.py
├── customer_churn.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
└── README.md
```
