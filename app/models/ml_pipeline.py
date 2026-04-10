"""
GlucoAI - ML Model Pipeline
Diabetes risk prediction using Random Forest + SHAP explainability.
Anomaly detection via Isolation Forest.
SHAP is optional — falls back to permutation-based proxy when not installed.
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).parent
DATA_DIR  = Path(__file__).parent.parent.parent / "data"
MODEL_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
FEATURE_COLS = [
    "age", "bmi", "glucose", "hba1c", "blood_pressure",
    "insulin", "skin_thickness", "pregnancies",
    "activity_level", "sleep_hours", "stress_level",
    "family_history", "smoker", "hypertension",
]
FEATURE_DISPLAY_NAMES = {
    "age": "Age", "bmi": "BMI", "glucose": "Glucose Level",
    "hba1c": "HbA1c", "blood_pressure": "Blood Pressure",
    "insulin": "Insulin", "skin_thickness": "Skin Thickness",
    "pregnancies": "Pregnancies", "activity_level": "Activity Level",
    "sleep_hours": "Sleep Hours", "stress_level": "Stress Level",
    "family_history": "Family History", "smoker": "Smoker",
    "hypertension": "Hypertension",
}


# ─────────────────────────────────────────────
# DATASET GENERATION
# ─────────────────────────────────────────────

def generate_synthetic_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age            = rng.integers(20, 80, n_samples).astype(float)
    bmi            = rng.normal(27, 5, n_samples).clip(15, 50)
    glucose        = rng.normal(120, 30, n_samples).clip(60, 300)
    hba1c          = rng.normal(6.0, 1.2, n_samples).clip(4.0, 12.0)
    blood_pressure = rng.normal(75, 12, n_samples).clip(50, 120)
    insulin        = rng.exponential(80, n_samples).clip(0, 400)
    skin_thickness = rng.normal(25, 10, n_samples).clip(5, 60)
    pregnancies    = rng.integers(0, 10, n_samples).astype(float)
    activity_level = rng.integers(0, 5, n_samples).astype(float)
    sleep_hours    = rng.normal(7, 1.2, n_samples).clip(4, 10)
    stress_level   = rng.integers(0, 3, n_samples).astype(float)
    family_history = rng.integers(0, 2, n_samples).astype(float)
    smoker         = rng.integers(0, 2, n_samples).astype(float)
    hypertension   = rng.integers(0, 2, n_samples).astype(float)

    log_odds = (
        -8.0 + 0.04*age + 0.10*bmi + 0.02*glucose + 0.50*hba1c
        + 0.005*blood_pressure + 0.003*insulin + 0.60*family_history
        + 0.40*hypertension - 0.25*activity_level - 0.15*sleep_hours
        + 0.20*stress_level + 0.30*smoker
    )
    prob = (1 / (1 + np.exp(-log_odds)) + rng.normal(0, 0.05, n_samples)).clip(0, 1)
    is_diabetic = (prob > 0.5).astype(int)

    return pd.DataFrame({
        "age": age, "bmi": bmi, "glucose": glucose, "hba1c": hba1c,
        "blood_pressure": blood_pressure, "insulin": insulin,
        "skin_thickness": skin_thickness, "pregnancies": pregnancies,
        "activity_level": activity_level, "sleep_hours": sleep_hours,
        "stress_level": stress_level, "family_history": family_history,
        "smoker": smoker, "hypertension": hypertension,
        "is_diabetic": is_diabetic,
    })


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col, (lo, hi) in {
        "bmi":(10,60),"glucose":(40,400),"hba1c":(3.5,15.0),
        "age":(0,110),"blood_pressure":(40,130),"insulin":(0,600),"sleep_hours":(2,12),
    }.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


def _permutation_shap_proxy(model, x_scaled: np.ndarray) -> np.ndarray:
    """Permutation-based feature importance as SHAP proxy."""
    base = model.predict_proba(x_scaled)[0, 1]
    approx = np.zeros(x_scaled.shape[1])
    for i in range(x_scaled.shape[1]):
        p = x_scaled.copy(); p[0, i] = 0.0
        approx[i] = base - model.predict_proba(p)[0, 1]
    return approx


# ─────────────────────────────────────────────
# RISK MODEL
# ─────────────────────────────────────────────

class DiabetesRiskModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=10,
            min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.explainer = None
        self.feature_importances_ = None
        self.version = "1.0.0"
        self.is_trained = False

    def train(self, df: pd.DataFrame) -> Dict:
        df = preprocess_features(df)
        X, y = df[FEATURE_COLS].values, df["is_diabetic"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        Xtr = self.scaler.fit_transform(X_train)
        Xte = self.scaler.transform(X_test)
        self.model.fit(Xtr, y_train)

        if SHAP_AVAILABLE:
            self.explainer = shap.TreeExplainer(self.model)

        self.feature_importances_ = dict(zip(FEATURE_COLS, self.model.feature_importances_.tolist()))
        y_pred = self.model.predict(Xte)
        y_prob = self.model.predict_proba(Xte)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        cv = cross_val_score(self.model, Xtr, y_train, cv=StratifiedKFold(5), scoring="roc_auc")
        self.is_trained = True
        return {
            "roc_auc": round(roc_auc, 4),
            "cv_auc_mean": round(cv.mean(), 4),
            "cv_auc_std": round(cv.std(), 4),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "n_train": len(y_train), "n_test": len(y_test),
            "shap_available": SHAP_AVAILABLE,
        }

    def predict(self, features: Dict) -> Dict:
        if not self.is_trained:
            self.load()
        x = np.array([[features.get(f, 0) for f in FEATURE_COLS]])
        xs = self.scaler.transform(x)
        prob = self.model.predict_proba(xs)[0, 1]
        risk_level = "high" if prob >= 0.65 else "medium" if prob >= 0.35 else "low"

        if SHAP_AVAILABLE and self.explainer is not None:
            sv = self.explainer.shap_values(xs)
            raw = sv[1][0] if isinstance(sv, list) else sv[0]
        else:
            raw = _permutation_shap_proxy(self.model, xs)

        shap_dict = {FEATURE_DISPLAY_NAMES.get(k, k): round(float(v), 5)
                     for k, v in zip(FEATURE_COLS, raw)}
        return {
            "risk_score": round(float(prob), 4),
            "risk_level": risk_level,
            "shap_values": shap_dict,
            "feature_importances": {
                FEATURE_DISPLAY_NAMES.get(k, k): round(v, 4)
                for k, v in self.feature_importances_.items()},
        }

    def save(self):
        with open(MODEL_DIR / "risk_model.pkl", "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "explainer": self.explainer,
                         "feature_importances": self.feature_importances_,
                         "version": self.version}, f)
        print(f"✅ Model saved → {MODEL_DIR / 'risk_model.pkl'}")

    def load(self):
        path = MODEL_DIR / "risk_model.pkl"
        if not path.exists():
            print("⚠️  No saved model — training now …")
            self.train(generate_synthetic_dataset())
            self.save()
            return
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.model = d["model"]; self.scaler = d["scaler"]
        self.explainer = d.get("explainer")
        self.feature_importances_ = d["feature_importances"]
        self.version = d.get("version", "?")
        self.is_trained = True
        print(f"✅ Model loaded (v{self.version})")


# ─────────────────────────────────────────────
# ANOMALY DETECTOR
# ─────────────────────────────────────────────

class GlucoseAnomalyDetector:
    HYPO = 70; HIGH = 180; CRIT_LOW = 54; CRIT_HIGH = 250

    def __init__(self):
        self.model = IsolationForest(contamination=0.05, n_estimators=100, random_state=42)
        self.scaler = StandardScaler(); self.is_fitted = False

    def _feats(self, readings):
        arr, w = np.array(readings), 3
        return np.array([[arr[i], arr[max(0,i-w):i+1].mean(),
                          arr[max(0,i-w):i+1].std() if i>0 else 0,
                          arr[i]-arr[i-1] if i>0 else 0]
                         for i in range(len(arr))])

    def fit(self, readings):
        F = self._feats(readings); self.scaler.fit(F)
        self.model.fit(self.scaler.transform(F)); self.is_fitted = True

    def detect(self, readings) -> List[Dict]:
        if not self.is_fitted: self.fit(readings)
        F = self.scaler.transform(self._feats(readings))
        scores, preds = self.model.score_samples(F), self.model.predict(F)
        out = []
        for i, (g, sc, pr) in enumerate(zip(readings, scores, preds)):
            atype = ("critical_low" if g < self.CRIT_LOW else
                     "hypoglycemia" if g < self.HYPO else
                     "critical_high" if g > self.CRIT_HIGH else
                     "hyperglycemia" if g > self.HIGH else
                     "pattern_anomaly" if pr == -1 else None)
            out.append({"index": i, "glucose": g,
                        "is_anomaly": (g < self.HYPO or g > self.HIGH or pr == -1),
                        "anomaly_score": round(float(sc), 4),
                        "anomaly_type": atype,
                        "is_critical": g < self.CRIT_LOW or g > self.CRIT_HIGH})
        return out


# ─────────────────────────────────────────────
# TREND ANALYZER
# ─────────────────────────────────────────────

class GlucoseTrendAnalyzer:
    @staticmethod
    def rolling_stats(readings, timestamps=None, window=4):
        df = pd.DataFrame({"glucose": readings})
        if timestamps: df["timestamp"] = pd.to_datetime(timestamps)
        df["rolling_mean"] = df["glucose"].rolling(window, min_periods=1).mean()
        df["rolling_std"]  = df["glucose"].rolling(window, min_periods=1).std().fillna(0)
        df["delta"]        = df["glucose"].diff().fillna(0)
        df["is_spike"]     = df["delta"].abs() > 50
        return df

    @staticmethod
    def time_in_range(readings):
        arr, n = np.array(readings), len(readings)
        return {
            "in_range_pct":    round(float(((arr>=70)&(arr<=180)).sum()/n*100), 1),
            "below_range_pct": round(float((arr<70).sum()/n*100), 1),
            "above_range_pct": round(float((arr>180).sum()/n*100), 1),
            "mean": round(float(arr.mean()),1), "std": round(float(arr.std()),1),
            "cv":   round(float(arr.std()/arr.mean()*100),1) if arr.mean()>0 else 0,
        }

    @staticmethod
    def weekly_summary(df):
        if "timestamp" not in df.columns: return []
        df = df.copy(); df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        d = df.groupby("date")["glucose"].agg(["mean","min","max","std"]).reset_index()
        d.columns = ["date","avg","min","max","std"]; d["date"] = d["date"].astype(str)
        return d.to_dict(orient="records")

    @staticmethod
    def trend_direction(readings, window=7):
        if len(readings) < 3: return "insufficient_data"
        r = readings[-window:] if len(readings) >= window else readings
        slope = np.polyfit(range(len(r)), r, 1)[0]
        return "worsening" if slope > 2 else "improving" if slope < -2 else "stable"


# ─────────────────────────────────────────────
# MEAL RECOMMENDER
# ─────────────────────────────────────────────

class MealRecommender:
    def __init__(self, gi_path=None):
        path = gi_path or str(DATA_DIR / "glycemic_index.csv")
        try:
            self.gi_db = pd.read_csv(path)
        except Exception:
            self.gi_db = self._fallback_db()

    @staticmethod
    def _fallback_db():
        return pd.DataFrame({
            "food_name":        ["White rice","Brown rice","White bread","Whole grain bread",
                                 "Apple","Banana","Oatmeal","Lentils","Salmon","Broccoli",
                                 "Potato","Chips","Almonds","Greek yogurt","Pizza","Burger"],
            "glycemic_index":   [72,50,75,51,36,51,55,32,0,10,82,54,0,11,60,66],
            "diabetes_friendly":[False,True,False,True,True,True,True,True,True,True,
                                 False,False,True,True,False,False],
            "alternatives":     ["Brown rice","","Whole grain bread","","","Berries","",
                                 "","","","Cauliflower mash","Nuts","","",
                                 "Cauliflower pizza","Grilled chicken salad"],
            "notes":            ["High GI","Good choice","Avoid","Good fiber","Low GI","Moderate",
                                 "Slow release","Excellent","Omega-3","Excellent","High GI","Avoid",
                                 "Great snack","Low GI","Limit","Avoid"],
        })

    def analyze_meal(self, meal_text, glucose_level=None):
        ml = meal_text.lower()
        mask = self.gi_db["food_name"].str.lower().apply(
            lambda n: any(w in ml for w in n.lower().split() if len(w) > 2))
        matched = self.gi_db[mask]
        if matched.empty:
            return {"matched_foods":[],"avg_glycemic_index":None,"glucose_impact":"unknown",
                    "overall_rating":"unknown","foods_to_avoid":[],
                    "safer_alternatives":["Leafy greens","Lean protein","Nuts","Legumes","Berries"],
                    "safe_foods_in_meal":[],"notes":["Food not found in database."]}
        avg_gi = matched["glycemic_index"].mean()
        bad = matched[matched["diabetes_friendly"]==False]
        safe = matched[matched["diabetes_friendly"]==True]
        alts = []
        for _, r in bad.iterrows():
            a = str(r.get("alternatives",""))
            if a and a != "nan": alts.extend(x.strip() for x in a.split(","))
        return {
            "matched_foods": matched["food_name"].tolist(),
            "avg_glycemic_index": round(avg_gi,1),
            "glucose_impact": "low" if avg_gi<55 else "moderate" if avg_gi<70 else "high",
            "overall_rating": "good" if avg_gi<50 else "moderate" if avg_gi<65 else "poor",
            "foods_to_avoid": bad["food_name"].tolist(),
            "safer_alternatives": list(dict.fromkeys(alts))[:5],
            "safe_foods_in_meal": safe["food_name"].tolist(),
            "notes": matched["notes"].dropna().tolist(),
        }

    def what_if_analysis(self, current_meal, proposed_meal):
        c = self.analyze_meal(current_meal); p = self.analyze_meal(proposed_meal)
        cgi = c.get("avg_glycemic_index") or 65; pgi = p.get("avg_glycemic_index") or 55
        diff = cgi - pgi
        return {"current_meal":c,"proposed_meal":p,"gi_difference":round(diff,1),
                "projected_glucose_change":round(diff*0.5,1),
                "recommendation":"Switch recommended ✅" if diff>10 else "Similar glycemic impact ⚠️"}


# ─────────────────────────────────────────────
# SINGLETONS
# ─────────────────────────────────────────────

_risk_singleton = None; _anomaly_singleton = None

def get_risk_model() -> DiabetesRiskModel:
    global _risk_singleton
    if _risk_singleton is None:
        _risk_singleton = DiabetesRiskModel(); _risk_singleton.load()
    return _risk_singleton

def get_anomaly_detector() -> GlucoseAnomalyDetector:
    global _anomaly_singleton
    if _anomaly_singleton is None:
        _anomaly_singleton = GlucoseAnomalyDetector()
    return _anomaly_singleton


if __name__ == "__main__":
    print("🔬 Training GlucoAI model …")
    df = generate_synthetic_dataset(3000)
    m  = DiabetesRiskModel(); metrics = m.train(df); m.save()
    print(f"ROC-AUC : {metrics['roc_auc']}")
    print(f"CV AUC  : {metrics['cv_auc_mean']} ± {metrics['cv_auc_std']}")
    print(f"SHAP    : {'enabled' if metrics['shap_available'] else 'proxy mode'}")
    print(); print(metrics["classification_report"])
