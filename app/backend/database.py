"""
GlucoAI - Database Models
SQLAlchemy ORM models for patient data, glucose readings, and predictions.
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

Base = declarative_base()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./glucoai.db")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), unique=True, index=True)
    name = Column(String(100))
    age = Column(Integer)
    gender = Column(String(10))
    weight_kg = Column(Float)
    height_cm = Column(Float)
    bmi = Column(Float)
    activity_level = Column(String(20))
    sleep_hours = Column(Float)
    stress_level = Column(String(10))
    hba1c = Column(Float, nullable=True)
    has_family_history = Column(Boolean, default=False)
    is_smoker = Column(Boolean, default=False)
    has_hypertension = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    glucose_readings = relationship("GlucoseReading", back_populates="patient")
    risk_predictions = relationship("RiskPrediction", back_populates="patient")
    meal_logs = relationship("MealLog", back_populates="patient")


class GlucoseReading(Base):
    __tablename__ = "glucose_readings"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"))
    glucose_level = Column(Float)
    reading_type = Column(String(20))  # fasting, post_meal, random
    meal_type = Column(String(20), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)

    patient = relationship("Patient", back_populates="glucose_readings")


class RiskPrediction(Base):
    __tablename__ = "risk_predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"))
    risk_score = Column(Float)
    risk_level = Column(String(10))  # low, medium, high
    model_version = Column(String(20))
    features_used = Column(Text)  # JSON string of features
    shap_values = Column(Text)     # JSON string of SHAP values
    ai_explanation = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="risk_predictions")


class MealLog(Base):
    __tablename__ = "meal_logs"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"))
    meal_description = Column(Text)
    meal_type = Column(String(20))
    glycemic_load_estimate = Column(Float, nullable=True)
    glucose_impact = Column(String(20), nullable=True)  # low, moderate, high
    recommendations = Column(Text, nullable=True)  # JSON string
    timestamp = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="meal_logs")


class TrendAnalysis(Base):
    __tablename__ = "trend_analyses"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50))
    period = Column(String(10))  # daily, weekly
    avg_glucose = Column(Float)
    min_glucose = Column(Float)
    max_glucose = Column(Float)
    std_glucose = Column(Float)
    spike_count = Column(Integer)
    hypoglycemia_count = Column(Integer)
    time_in_range = Column(Float)  # percentage
    trend_direction = Column(String(10))  # improving, stable, worsening
    analysis_date = Column(DateTime, default=datetime.utcnow)


def get_engine():
    return create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})


def get_session():
    engine = get_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def init_db():
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully")
