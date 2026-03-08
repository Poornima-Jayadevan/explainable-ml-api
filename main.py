from typing import Any, Dict, List
import numpy as np
import pandas as pd
import joblib
import shap

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from io import StringIO

import io
import matplotlib
matplotlib.use("Agg")  # important for server environments (no GUI)
import matplotlib.pyplot as plt

from fastapi.responses import StreamingResponse

MODEL_PATH = "risk_model.pkl"
FEATURES_PATH = "feature_names.pkl"

model = None
feature_names: List[str] = []
explainer = None

def ffloat(x: Any, ndigits: int = 6) -> float:
    """Convert to float safely and round."""
    return float(np.round(float(np.ravel(x)[0]), ndigits))

def clean_feature_name(feat: str) -> str:
    """Optional: make one-hot names easier to read."""
    return feat.replace("_", " ")

def format_contribs(
    cols: List[str],
    vals: np.ndarray,
    shap_vec: np.ndarray,
    top_k: int = 5,
    ndigits: int = 6
) -> Dict[str, Any]:
    """
    Returns a clean explanation payload:
    - top_contributions (abs sorted)
    - top_positive (increases risk)
    - top_negative (decreases risk)
    """
    records = []
    for feat, val, sv in zip(cols, vals, shap_vec):
        impact = ffloat(sv, ndigits)
        value = ffloat(val, ndigits)
        records.append({
            "feature": feat,  # or clean_feature_name(feat)
            "value": value,
            "impact": impact,  # SHAP value
            "direction": "increases_risk" if impact > 0 else "decreases_risk"
        })

    # Sort by absolute impact
    records_sorted = sorted(records, key=lambda r: abs(r["impact"]), reverse=True)

    top = records_sorted[:top_k]
    top_pos = [r for r in records_sorted if r["impact"] > 0][:top_k]
    top_neg = [r for r in records_sorted if r["impact"] < 0][:top_k]

    return {
        "top_contributions": top,
        "top_positive": top_pos,
        "top_negative": top_neg
    }

class RiskInput(BaseModel):
    # numeric
    person_age: float
    person_income: float
    person_emp_length: float
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float

    # categorical
    person_home_ownership: str
    loan_intent: str
    loan_grade: str
    cb_person_default_on_file: str


def preprocess(payload: RiskInput) -> pd.DataFrame:
    raw = pd.DataFrame([payload.model_dump()])

    categorical_cols = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]

    encoded = pd.get_dummies(raw, columns=categorical_cols, drop_first=True)
    encoded = encoded.reindex(columns=feature_names, fill_value=0)
    return encoded


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_names, explainer
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    explainer = shap.TreeExplainer(model)
    yield


app = FastAPI(
    title="Explainable Risk Classification API",
    description="Machine Learning API for loan risk prediction with SHAP explainability",
    version="1.0",
    lifespan=lifespan
)


# CORS configuration (allow frontend apps to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (good for development)
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"],  # allow all headers
)


@app.get("/")
def home() -> Dict[str, str]:
    return {"message": "Explainable ML API is running"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "n_features": len(feature_names),
    }


@app.get("/features")
def features() -> Dict[str, Any]:
    return {"feature_names": feature_names}


@app.post("/predict")
def predict(payload: RiskInput) -> Dict[str, Any]:
    X = preprocess(payload)

    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None

    return {
        "prediction": pred,
        "probability_class_1": proba
    }


@app.post("/explain")
def explain(payload: RiskInput) -> Dict[str, Any]:
    try:
        X = preprocess(payload)

        exp = explainer(X)
        values = exp.values
        base_values = exp.base_values

        # --- pick class 1 if (n, f, c) ---
        if values.ndim == 3:
            class_idx = 1
            shap_vec = values[0, :, class_idx]
            base_val = base_values[0, class_idx]
        else:
            shap_vec = values[0]
            base_val = base_values[0]

        shap_vec = np.asarray(shap_vec).reshape(-1)
        vals = X.iloc[0].to_numpy().reshape(-1)
        cols = list(X.columns)

        if len(shap_vec) != len(vals) or len(vals) != len(cols):
            raise ValueError(
                f"Length mismatch: shap={len(shap_vec)}, vals={len(vals)}, cols={len(cols)}"
            )

        # --- formatting helpers ---
        def r(x: Any, nd: int = 6) -> float:
            return float(np.round(float(np.ravel(x)[0]), nd))

        # --- build records ---
        records: List[Dict[str, Any]] = []
        for feat, val, sv in zip(cols, vals, shap_vec):
            sv_f = r(sv, 6)
            val_f = r(val, 6)

            records.append({
                "feature": feat,
                "value": val_f,
                "impact": sv_f,  # renamed from shap_value -> impact (cleaner)
                "abs_impact": abs(sv_f),
                "direction": "increases_risk" if sv_f > 0 else "decreases_risk",
            })

        # --- sort and slice ---
        records_sorted = sorted(records, key=lambda r: r["abs_impact"], reverse=True)

        top_contributions = [
            {
                "feature": rec["feature"],
                "value": rec["value"],
                "impact": rec["impact"],
                "direction": rec["direction"],
            }
            for rec in records_sorted[:10]
        ]

        top_positive = [
            {
                "feature": rec["feature"],
                "value": rec["value"],
                "impact": rec["impact"],
            }
            for rec in records_sorted
            if rec["impact"] > 0
        ][:5]

        top_negative = [
            {
                "feature": rec["feature"],
                "value": rec["value"],
                "impact": rec["impact"],
            }
            for rec in records_sorted
            if rec["impact"] < 0
        ][:5]

        # --- model outputs ---
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None

        # Sort by abs impact (already have records_sorted)
        records_sorted = sorted(records, key=lambda rec: rec["abs_impact"], reverse=True)

        # Keep top 10 for display
        top_n = 10
        top_records = records_sorted[:top_n]

        # Aggregate the rest into "other" (optional but nice for waterfall)
        other_impact = sum(rec["impact"] for rec in records_sorted[top_n:])
        if abs(other_impact) > 0:
            top_records = top_records + [{
                "feature": "other",
                "value": 0.0,
                "impact": float(np.round(other_impact, 6)),
                "abs_impact": abs(float(np.round(other_impact, 6))),
                "direction": "increases_risk" if other_impact > 0 else "decreases_risk",
            }]

        # Build waterfall steps
        start = float(np.round(float(np.ravel(base_val)[0]), 6))
        running = start
        steps = []
        for rec in top_records:
            before = running
            after = float(np.round(before + rec["impact"], 6))
            steps.append({
                "feature": rec["feature"],
                "impact": rec["impact"],
                "direction": rec["direction"],
                "value_before": before,
                "value_after": after,
            })
            running = after

        final_value = running

        return {
            "prediction": pred,
            "risk_score": r(proba, 6) if proba is not None else None,
            "base_value": start,

            # existing pretty list (optional keep)
            "top_contributions": [
                {
                    "feature": rec["feature"],
                    "value": rec["value"],
                    "impact": rec["impact"],
                    "direction": rec["direction"],
                }
                for rec in records_sorted[:10]
            ],

            # ✅ waterfall-ready payload
            "waterfall": {
                "start": start,
                "final": final_value,
                "steps": steps
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a CSV with the SAME raw columns as RiskInput:
    numeric columns + categorical columns (strings).
    Returns predictions + probabilities for all rows.
    """
    try:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Please upload a .csv file")

        content = await file.read()
        df_raw = pd.read_csv(StringIO(content.decode("utf-8")))

        # Required raw columns (same as RiskInput)
        required_cols = [
            "person_age",
            "person_income",
            "person_emp_length",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
            "person_home_ownership",
            "loan_intent",
            "loan_grade",
            "cb_person_default_on_file",
        ]

        missing = [c for c in required_cols if c not in df_raw.columns]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"CSV missing required columns: {missing}"
            )

        # One-hot encode like training (drop_first=True)
        categorical_cols = [
            "person_home_ownership",
            "loan_intent",
            "loan_grade",
            "cb_person_default_on_file",
        ]
        X_encoded = pd.get_dummies(df_raw[required_cols], columns=categorical_cols, drop_first=True)

        # Align to training features
        X_encoded = X_encoded.reindex(columns=feature_names, fill_value=0)

        # Predict
        preds = model.predict(X_encoded).astype(int).tolist()
        probas = None
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_encoded)[:, 1].astype(float).tolist()

        return {
            "n_rows": int(len(df_raw)),
            "predictions": preds,
            "probability_class_1": probas,
        }

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Could not decode file. Please upload a UTF-8 CSV.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/batch_explain")
async def batch_explain(
    file: UploadFile = File(...),
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Upload a CSV with the SAME raw columns as RiskInput.
    Returns predictions + risk_score + top_k SHAP contributions per row.
    """
    try:
        if top_k < 1 or top_k > 20:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Please upload a .csv file")

        content = await file.read()
        df_raw = pd.read_csv(StringIO(content.decode("utf-8")))

        required_cols = [
            "person_age",
            "person_income",
            "person_emp_length",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
            "person_home_ownership",
            "loan_intent",
            "loan_grade",
            "cb_person_default_on_file",
        ]

        missing = [c for c in required_cols if c not in df_raw.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"CSV missing required columns: {missing}")

        categorical_cols = [
            "person_home_ownership",
            "loan_intent",
            "loan_grade",
            "cb_person_default_on_file",
        ]

        X_encoded = pd.get_dummies(df_raw[required_cols], columns=categorical_cols, drop_first=True)
        X_encoded = X_encoded.reindex(columns=feature_names, fill_value=0)

        # Predictions
        preds = model.predict(X_encoded).astype(int)
        probas = model.predict_proba(X_encoded)[:, 1].astype(float) if hasattr(model, "predict_proba") else None

        # SHAP explanations for ALL rows at once
        exp = explainer(X_encoded)
        values = exp.values
        base_values = exp.base_values

        # Handle shapes: (n,f) or (n,f,c)
        if values.ndim == 3:
            class_idx = 1
            shap_vals = values[:, :, class_idx]
            base_vals = base_values[:, class_idx]
        else:
            shap_vals = values
            base_vals = base_values

        cols = list(X_encoded.columns)

        # rounding helper
        def r(x: Any, nd: int = 6) -> float:
            return float(np.round(float(np.ravel(x)[0]), nd))

        results: List[Dict[str, Any]] = []

        for i in range(len(X_encoded)):
            row_vals = X_encoded.iloc[i].to_numpy().reshape(-1)
            row_shap = np.asarray(shap_vals[i]).reshape(-1)

            # Build records
            records: List[Dict[str, Any]] = []
            for feat, val, sv in zip(cols, row_vals, row_shap):
                impact = r(sv, 6)
                value = r(val, 6)

                records.append({
                    "feature": feat,
                    "value": value,
                    "impact": impact,
                    "abs_impact": abs(impact),
                    "direction": "increases_risk" if impact > 0 else "decreases_risk"
                })

            # Sort by absolute impact
            records_sorted = sorted(records, key=lambda rec: rec["abs_impact"], reverse=True)

            # --- waterfall (top_k + other) ---
            top_records = records_sorted[:top_k]
            other_impact = sum(rec["impact"] for rec in records_sorted[top_k:])
            if abs(other_impact) > 0:
                top_records = top_records + [{
                    "feature": "other",
                    "value": 0.0,
                    "impact": float(np.round(other_impact, 6)),
                    "abs_impact": abs(float(np.round(other_impact, 6))),
                    "direction": "increases_risk" if other_impact > 0 else "decreases_risk",
                }]

            start = r(base_vals[i], 6)
            running = start
            steps = []
            for rec in top_records:
                before = running
                after = float(np.round(before + rec["impact"], 6))
                steps.append({
                    "feature": rec["feature"],
                    "impact": rec["impact"],
                    "direction": rec["direction"],
                    "value_before": before,
                    "value_after": after,
                })
                running = after


            top_contributions = [
                {
                    "feature": rec["feature"],
                    "value": rec["value"],
                    "impact": rec["impact"],
                    "direction": rec["direction"],
                }
                for rec in records_sorted[:top_k]
            ]

            top_positive = [
                {"feature": rec["feature"], "value": rec["value"], "impact": rec["impact"]}
                for rec in records_sorted if rec["impact"] > 0
            ][:top_k]

            top_negative = [
                {"feature": rec["feature"], "value": rec["value"], "impact": rec["impact"]}
                for rec in records_sorted if rec["impact"] < 0
            ][:top_k]

            results.append({
                "row_id": int(i),
                "prediction": int(preds[i]),
                "risk_score": r(probas[i], 6) if probas is not None else None,
                "base_value": start,
                "top_contributions": top_contributions,
                "top_positive": top_positive,
                "top_negative": top_negative,

                # ✅ waterfall-ready
                "waterfall": {
                    "start": start,
                    "final": running,
                    "steps": steps
                }
            })

        return {
            "n_rows": int(len(df_raw)),
            "top_k": int(top_k),
            "results": results
        }

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Could not decode file. Please upload a UTF-8 CSV.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain_waterfall_png")
def explain_waterfall_png(payload: RiskInput) -> StreamingResponse:
    """
    Returns a SHAP waterfall plot as a PNG image for ONE row (the given payload).
    """
    X = preprocess(payload)

    # Compute SHAP Explanation
    exp = explainer(X)

    # Handle (n, f) vs (n, f, c)
    # We need a single-row, single-class Explanation for the plot.
    values = exp.values
    base_values = exp.base_values

    if values.ndim == 3:
        class_idx = 1
        row_exp = shap.Explanation(
            values=values[0, :, class_idx],
            base_values=base_values[0, class_idx],
            data=X.iloc[0].to_numpy(),
            feature_names=list(X.columns),
        )
    else:
        row_exp = shap.Explanation(
            values=values[0],
            base_values=base_values[0],
            data=X.iloc[0].to_numpy(),
            feature_names=list(X.columns),
        )

    # Build the plot (matplotlib)
    plt.figure()
    shap.plots.waterfall(row_exp, max_display=10, show=False)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    plt.close()
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")