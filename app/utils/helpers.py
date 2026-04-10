"""
GlucoAI - Utility Functions
Groq/LLM integration, formatting helpers, and clinical utilities.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any


# ─────────────────────────────────────────────
# GROQ / LLM INTEGRATION
# ─────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-8b-8192"


def call_groq(prompt: str, system_prompt: str = "", max_tokens: int = 512) -> str:
    """
    Call Groq API with Llama3 and return the assistant message.
    Falls back to rule-based explanation if API key is missing.
    """
    if not GROQ_API_KEY:
        return _rule_based_explanation(prompt)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.5,
    }

    try:
        resp = requests.post(GROQ_BASE_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return _rule_based_explanation(prompt)


def _rule_based_explanation(context: str) -> str:
    """Fallback explanation without LLM."""
    return (
        "Based on your health data, our AI model has analyzed your glucose patterns and risk factors. "
        "Key findings are highlighted in the SHAP chart above — bars pointing right increase risk, "
        "bars pointing left decrease risk. For personalized advice, consult a healthcare professional. "
        "(Note: Configure GROQ_API_KEY for AI-powered natural language explanations.)"
    )


def generate_risk_explanation(
    risk_level: str,
    risk_score: float,
    shap_values: Dict[str, float],
    patient_features: Dict[str, Any],
) -> str:
    """Generate a patient-friendly explanation of the risk prediction."""
    top_factors = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
    top_risk_factors = [(k, v) for k, v in top_factors if v > 0]
    top_protective = [(k, v) for k, v in top_factors if v < 0]

    factors_text = ", ".join([f"{k} (↑ risk)" for k, _ in top_risk_factors[:3]])
    protective_text = ", ".join([f"{k} (↓ risk)" for k, _ in top_protective[:2]])

    system = (
        "You are a compassionate, expert medical AI assistant. Explain diabetes risk predictions "
        "in simple, supportive language. Be encouraging, never alarmist. Keep responses under 150 words. "
        "Focus on actionable steps."
    )

    prompt = f"""
A patient has a diabetes risk score of {risk_score:.0%} ({risk_level.upper()} risk).

Top risk factors: {factors_text or 'none identified'}
Protective factors: {protective_text or 'none identified'}

Patient profile:
- Age: {patient_features.get('age', 'N/A')} years
- BMI: {patient_features.get('bmi', 'N/A'):.1f} kg/m²
- Activity: {patient_features.get('activity_level', 'N/A')} (0=sedentary, 4=very active)

Write 3 sentences: (1) explain what this risk score means, (2) highlight the main contributing factor, 
(3) give one specific actionable recommendation.
"""
    return call_groq(prompt, system_prompt=system, max_tokens=200)


def generate_glucose_insights(
    readings: List[float],
    anomalies: List[Dict],
    trend_direction: str,
    tir: Dict,
) -> str:
    """Generate AI-powered glucose trend insights."""
    anomaly_count = len([a for a in anomalies if a.get("is_anomaly")])
    system = (
        "You are a glucose monitoring specialist AI. Provide concise, actionable insights "
        "about glucose patterns. Always end with a motivational tip. Keep it under 120 words."
    )
    prompt = f"""
Glucose analysis for patient:
- Readings: {len(readings)} data points
- Average: {sum(readings)/len(readings):.1f} mg/dL
- Range: {min(readings):.0f}–{max(readings):.0f} mg/dL
- Time in range (70-180): {tir.get('in_range_pct', 'N/A')}%
- Trend: {trend_direction}
- Anomalies detected: {anomaly_count}

Provide a 3-sentence summary: (1) overall glucose control assessment, 
(2) most notable pattern or concern, (3) top recommendation.
"""
    return call_groq(prompt, system_prompt=system, max_tokens=180)


def generate_meal_advice(
    meal_description: str,
    analysis: Dict,
    current_glucose: Optional[float] = None,
) -> str:
    """Generate personalized meal advice using LLM."""
    system = (
        "You are a registered dietitian specializing in diabetes management. "
        "Give practical, specific, culturally sensitive meal advice. Under 100 words."
    )
    prompt = f"""
Meal logged: "{meal_description}"
Glycemic index: {analysis.get('avg_glycemic_index', 'unknown')}
Glucose impact: {analysis.get('glucose_impact', 'unknown')}
Current glucose: {current_glucose or 'not provided'} mg/dL
Foods to avoid: {', '.join(analysis.get('foods_to_avoid', [])[:3]) or 'none identified'}
Alternatives: {', '.join(analysis.get('safer_alternatives', [])[:3]) or 'see general guidelines'}

Give 2 specific suggestions to make this meal more diabetes-friendly.
"""
    return call_groq(prompt, system_prompt=system, max_tokens=150)


# ─────────────────────────────────────────────
# CLINICAL REFERENCE UTILITIES
# ─────────────────────────────────────────────

def classify_glucose(value: float, context: str = "random") -> Dict:
    """Classify a glucose reading according to ADA guidelines."""
    if context == "fasting":
        if value < 70:
            return {"category": "hypoglycemia", "color": "#ef4444", "severity": "critical"}
        elif value < 100:
            return {"category": "normal", "color": "#22c55e", "severity": "none"}
        elif value < 126:
            return {"category": "prediabetes", "color": "#f59e0b", "severity": "warning"}
        else:
            return {"category": "diabetes_range", "color": "#ef4444", "severity": "high"}
    else:
        if value < 70:
            return {"category": "hypoglycemia", "color": "#ef4444", "severity": "critical"}
        elif value <= 140:
            return {"category": "normal", "color": "#22c55e", "severity": "none"}
        elif value <= 199:
            return {"category": "prediabetes", "color": "#f59e0b", "severity": "warning"}
        else:
            return {"category": "diabetes_range", "color": "#ef4444", "severity": "high"}


def estimate_hba1c_from_glucose(avg_glucose: float) -> float:
    """Nathan formula: HbA1c = (avg_glucose + 46.7) / 28.7"""
    return round((avg_glucose + 46.7) / 28.7, 1)


def get_glucose_recommendations(reading: float, context: str = "random") -> List[str]:
    """Return immediate action recommendations based on glucose level."""
    if reading < 54:
        return [
            "🚨 EMERGENCY: Consume 15–20g fast-acting carbs immediately",
            "Call emergency services if unconscious or unable to swallow",
            "Recheck in 15 minutes",
        ]
    elif reading < 70:
        return [
            "⚠️ Take 15g fast-acting carbs (4 glucose tablets, 4oz juice)",
            "Recheck glucose in 15 minutes",
            "Avoid exercise until glucose is above 100 mg/dL",
        ]
    elif reading <= 180:
        return [
            "✅ Glucose is in a healthy range",
            "Maintain current routine and diet",
        ]
    elif reading <= 250:
        return [
            "⚠️ Elevated — avoid high-carb foods for next 2 hours",
            "Light walking may help lower glucose",
            "Monitor again in 1 hour",
        ]
    else:
        return [
            "🚨 Very high glucose — contact your healthcare provider",
            "Stay well hydrated with water",
            "Do not exercise with ketones present",
            "Check for illness or medication issues",
        ]


def activity_level_to_numeric(activity: str) -> float:
    mapping = {
        "sedentary": 0, "light": 1, "moderate": 2, "active": 3, "very_active": 4
    }
    return mapping.get(activity.lower(), 2)


def stress_level_to_numeric(stress: str) -> float:
    mapping = {"low": 0, "moderate": 1, "high": 2}
    return mapping.get(stress.lower(), 1)
