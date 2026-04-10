"""
GlucoAI - FastAPI Backend
Production-grade REST API for glucose monitoring and diabetes risk analysis.
"""

import json
import traceback
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ml_pipeline import (
    get_risk_model, get_anomaly_detector, GlucoseTrendAnalyzer,
    MealRecommender, DiabetesRiskModel, generate_synthetic_dataset
)
from backend.database import (
    init_db, get_session, Patient, GlucoseReading,
    RiskPrediction, MealLog, TrendAnalysis
)

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="GlucoAI API",
    description="AI-powered glucose monitoring and diabetes risk analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB and models on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    get_risk_model()   # warm model cache
    print("🚀 GlucoAI API started")


# ─────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────

class PatientCreate(BaseModel):
    patient_id: str = Field(..., example="P001")
    name: str = Field(..., example="Jane Doe")
    age: int = Field(..., ge=1, le=120)
    gender: str = Field(..., example="female")
    weight_kg: float = Field(..., gt=0)
    height_cm: float = Field(..., gt=0)
    activity_level: str = Field(default="moderate", example="moderate")
    sleep_hours: float = Field(default=7.0, ge=0, le=24)
    stress_level: str = Field(default="low")
    hba1c: Optional[float] = None
    has_family_history: bool = False
    is_smoker: bool = False
    has_hypertension: bool = False

    @validator("activity_level")
    def validate_activity(cls, v):
        valid = ["sedentary", "light", "moderate", "active", "very_active"]
        if v not in valid:
            raise ValueError(f"activity_level must be one of {valid}")
        return v


class GlucoseInput(BaseModel):
    patient_id: str
    glucose_level: float = Field(..., ge=20, le=600)
    reading_type: str = Field(default="random")
    meal_type: Optional[str] = None
    timestamp: Optional[str] = None
    notes: Optional[str] = None


class RiskPredictionRequest(BaseModel):
    patient_id: Optional[str] = None
    age: float = Field(..., ge=0, le=120)
    bmi: float = Field(..., ge=10, le=60)
    glucose: Optional[float] = Field(default=None)
    hba1c: Optional[float] = None
    blood_pressure: Optional[float] = None
    insulin: Optional[float] = None
    skin_thickness: Optional[float] = None
    pregnancies: Optional[float] = Field(default=0)
    activity_level: float = Field(default=2, ge=0, le=4)
    sleep_hours: float = Field(default=7.0)
    stress_level: float = Field(default=1, ge=0, le=2)
    family_history: float = Field(default=0)
    smoker: float = Field(default=0)
    hypertension: float = Field(default=0)


class MealAnalysisRequest(BaseModel):
    patient_id: Optional[str] = None
    meal_description: str
    meal_type: str = Field(default="lunch")
    current_glucose: Optional[float] = None


class WhatIfRequest(BaseModel):
    current_meal: str
    proposed_meal: str


class GlucoseBatchInput(BaseModel):
    patient_id: str
    readings: List[float]
    timestamps: Optional[List[str]] = None


# ─────────────────────────────────────────────
# HELPER: Build feature dict with defaults
# ─────────────────────────────────────────────

def build_features(req: RiskPredictionRequest) -> Dict:
    bmi = req.bmi
    glucose_estimate = req.glucose or (110 + bmi * 1.5)  # fallback estimate
    return {
        "age": req.age,
        "bmi": bmi,
        "glucose": glucose_estimate,
        "hba1c": req.hba1c or (5.0 + bmi * 0.05),
        "blood_pressure": req.blood_pressure or 75.0,
        "insulin": req.insulin or 80.0,
        "skin_thickness": req.skin_thickness or 25.0,
        "pregnancies": req.pregnancies or 0,
        "activity_level": req.activity_level,
        "sleep_hours": req.sleep_hours,
        "stress_level": req.stress_level,
        "family_history": req.family_history,
        "smoker": req.smoker,
        "hypertension": req.hypertension,
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {"status": "healthy", "service": "GlucoAI API", "version": "1.0.0"}


@app.get("/health", tags=["health"])
def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": get_risk_model().is_trained,
    }


# ── Patient Management ──

@app.post("/patients/", status_code=status.HTTP_201_CREATED, tags=["patients"])
def create_patient(patient: PatientCreate):
    """Register a new patient profile."""
    db = get_session()
    try:
        existing = db.query(Patient).filter(Patient.patient_id == patient.patient_id).first()
        if existing:
            raise HTTPException(status_code=409, detail="Patient already exists")
        bmi = patient.weight_kg / ((patient.height_cm / 100) ** 2)
        db_patient = Patient(
            **patient.dict(exclude={"weight_kg", "height_cm"}),
            weight_kg=patient.weight_kg,
            height_cm=patient.height_cm,
            bmi=round(bmi, 2),
        )
        db.add(db_patient)
        db.commit()
        db.refresh(db_patient)
        return {"message": "Patient created", "patient_id": patient.patient_id, "bmi": round(bmi, 2)}
    finally:
        db.close()


@app.get("/patients/{patient_id}", tags=["patients"])
def get_patient(patient_id: str):
    """Retrieve patient profile."""
    db = get_session()
    try:
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        return {
            "patient_id": patient.patient_id,
            "name": patient.name,
            "age": patient.age,
            "bmi": patient.bmi,
            "activity_level": patient.activity_level,
            "hba1c": patient.hba1c,
            "has_family_history": patient.has_family_history,
            "has_hypertension": patient.has_hypertension,
        }
    finally:
        db.close()


# ── Glucose Readings ──

@app.post("/glucose/", tags=["glucose"])
def log_glucose(reading: GlucoseInput):
    """Log a single glucose reading."""
    db = get_session()
    try:
        ts = datetime.fromisoformat(reading.timestamp) if reading.timestamp else datetime.utcnow()
        detector = get_anomaly_detector()
        anomaly_result = {"is_anomaly": False, "anomaly_score": 0}
        if not detector.is_fitted:
            # Use the single reading context
            anomaly_result = detector.detect([reading.glucose_level])[0]
        else:
            anomaly_result = detector.detect([reading.glucose_level])[0]

        db_reading = GlucoseReading(
            patient_id=reading.patient_id,
            glucose_level=reading.glucose_level,
            reading_type=reading.reading_type,
            meal_type=reading.meal_type,
            timestamp=ts,
            is_anomaly=anomaly_result.get("is_anomaly", False),
            anomaly_score=anomaly_result.get("anomaly_score"),
            notes=reading.notes,
        )
        db.add(db_reading)
        db.commit()

        alert = None
        if reading.glucose_level < 54:
            alert = {"level": "critical", "message": "⚠️ CRITICAL LOW — seek immediate help!"}
        elif reading.glucose_level < 70:
            alert = {"level": "warning", "message": "Low glucose — consume fast-acting carbs"}
        elif reading.glucose_level > 250:
            alert = {"level": "critical", "message": "⚠️ CRITICAL HIGH — consider contacting physician"}
        elif reading.glucose_level > 180:
            alert = {"level": "warning", "message": "Elevated glucose — monitor closely"}

        return {
            "message": "Glucose logged",
            "reading_id": db_reading.id,
            "anomaly_detected": anomaly_result.get("is_anomaly", False),
            "anomaly_type": anomaly_result.get("anomaly_type"),
            "alert": alert,
        }
    finally:
        db.close()


@app.post("/glucose/batch/", tags=["glucose"])
def analyze_glucose_batch(batch: GlucoseBatchInput):
    """Analyze a batch of glucose readings — anomalies + trends."""
    detector = get_anomaly_detector()
    anomalies = detector.detect(batch.readings)

    trend_df = GlucoseTrendAnalyzer.rolling_stats(
        batch.readings, batch.timestamps
    )
    tir = GlucoseTrendAnalyzer.time_in_range(batch.readings)
    direction = GlucoseTrendAnalyzer.trend_direction(batch.readings)

    spike_count = int(trend_df["is_spike"].sum())
    anomaly_count = sum(1 for a in anomalies if a["is_anomaly"])

    return {
        "patient_id": batch.patient_id,
        "n_readings": len(batch.readings),
        "time_in_range": tir,
        "trend_direction": direction,
        "spike_count": spike_count,
        "anomaly_count": anomaly_count,
        "anomalies": [a for a in anomalies if a["is_anomaly"]],
        "rolling_stats": trend_df[["glucose", "rolling_mean", "rolling_std", "delta", "is_spike"]].to_dict(orient="records"),
    }


@app.get("/glucose/{patient_id}/history", tags=["glucose"])
def get_glucose_history(patient_id: str, days: int = 7):
    """Retrieve glucose history for a patient."""
    db = get_session()
    try:
        since = datetime.utcnow() - timedelta(days=days)
        readings = (
            db.query(GlucoseReading)
            .filter(
                GlucoseReading.patient_id == patient_id,
                GlucoseReading.timestamp >= since
            )
            .order_by(GlucoseReading.timestamp)
            .all()
        )
        return {
            "patient_id": patient_id,
            "days": days,
            "readings": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "glucose_level": r.glucose_level,
                    "reading_type": r.reading_type,
                    "is_anomaly": r.is_anomaly,
                    "anomaly_type": None,
                }
                for r in readings
            ]
        }
    finally:
        db.close()


# ── Risk Prediction ──

@app.post("/predict/risk/", tags=["prediction"])
def predict_risk(request: RiskPredictionRequest):
    """Predict diabetes risk with SHAP explanation."""
    try:
        model = get_risk_model()
        features = build_features(request)
        result = model.predict(features)

        # Persist prediction if patient_id given
        if request.patient_id:
            db = get_session()
            try:
                pred = RiskPrediction(
                    patient_id=request.patient_id,
                    risk_score=result["risk_score"],
                    risk_level=result["risk_level"],
                    model_version=model.version,
                    features_used=json.dumps(features),
                    shap_values=json.dumps(result["shap_values"]),
                )
                db.add(pred)
                db.commit()
            finally:
                db.close()

        risk_messages = {
            "low": "Your risk profile is currently low. Maintain your healthy lifestyle!",
            "medium": "Moderate risk detected. Consider lifestyle modifications and regular check-ups.",
            "high": "High risk detected. Please consult a healthcare professional promptly.",
        }

        return {
            **result,
            "message": risk_messages[result["risk_level"]],
            "features_used": features,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Meal Analysis ──

@app.post("/meals/analyze/", tags=["meals"])
def analyze_meal(request: MealAnalysisRequest):
    """Analyze a meal and provide glucose impact + recommendations."""
    recommender = MealRecommender()
    analysis = recommender.analyze_meal(request.meal_description, request.current_glucose)

    if request.patient_id:
        db = get_session()
        try:
            log = MealLog(
                patient_id=request.patient_id,
                meal_description=request.meal_description,
                meal_type=request.meal_type,
                glycemic_load_estimate=analysis.get("avg_glycemic_index"),
                glucose_impact=analysis.get("glucose_impact"),
                recommendations=json.dumps(analysis.get("safer_alternatives", [])),
            )
            db.add(log)
            db.commit()
        finally:
            db.close()

    return analysis


@app.post("/meals/what-if/", tags=["meals"])
def what_if_analysis(request: WhatIfRequest):
    """Compare two meals and project glucose improvement."""
    recommender = MealRecommender()
    return recommender.what_if_analysis(request.current_meal, request.proposed_meal)


# ── Trend Analysis ──

@app.post("/trends/weekly/", tags=["trends"])
def compute_weekly_trends(batch: GlucoseBatchInput):
    """Compute weekly glucose trend summary."""
    if not batch.timestamps:
        return {"error": "timestamps required for weekly analysis"}
    import pandas as pd
    df = pd.DataFrame({"glucose": batch.readings, "timestamp": batch.timestamps})
    summary = GlucoseTrendAnalyzer.weekly_summary(df)
    tir = GlucoseTrendAnalyzer.time_in_range(batch.readings)
    direction = GlucoseTrendAnalyzer.trend_direction(batch.readings)
    return {
        "weekly_summary": summary,
        "time_in_range": tir,
        "trend_direction": direction,
    }


# ── Model Info ──

@app.get("/model/info/", tags=["model"])
def model_info():
    """Return model metadata and feature importances."""
    model = get_risk_model()
    return {
        "version": model.version,
        "is_trained": model.is_trained,
        "features": list(model.feature_importances_.keys()) if model.feature_importances_ else [],
        "feature_importances": model.feature_importances_,
    }


@app.post("/model/retrain/", tags=["model"])
def retrain_model(n_samples: int = 3000):
    """Retrain the risk model on fresh synthetic data."""
    from models.ml_pipeline import generate_synthetic_dataset, DiabetesRiskModel
    df = generate_synthetic_dataset(n_samples=n_samples)
    model = DiabetesRiskModel()
    metrics = model.train(df)
    model.save()
    # Reset singleton
    global _risk_model_singleton
    import app.models.ml_pipeline as mp
    mp._risk_model_singleton = model
    return {"message": "Model retrained", "metrics": metrics}
