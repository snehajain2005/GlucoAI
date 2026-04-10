# 🩺 GlucoAI — Smart Glucose Monitoring & Diabetes Risk Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)](https://scikit-learn.org/)

A production-grade AI-powered healthcare system for glucose monitoring, diabetes risk prediction, and personalized dietary recommendations.

---

## 🎯 Core Capabilities

| Feature | Description |
|---------|-------------|
| 📊 Glucose Tracking | Manual + API logging with real-time classification |
| 🔮 Risk Prediction | Random Forest model with SHAP explainability |
| 🧠 Anomaly Detection | Isolation Forest + clinical threshold rules |
| 📈 Trend Analysis | Rolling averages, spike detection, Time-in-Range |
| 🍽️ Meal Advisor | Glycemic index database with AI recommendations |
| ⚗️ What-If Analysis | Simulate diet & lifestyle change projections |
| 💬 AI Explanations | Groq Llama3 natural language insights |
| 🌐 Rural Mode | Simplified low-bandwidth UI |

---

## 🏗️ Architecture

```
glucoai/
├── app/
│   ├── backend/
│   │   ├── api.py           # FastAPI REST endpoints
│   │   └── database.py      # SQLAlchemy ORM models
│   ├── models/
│   │   └── ml_pipeline.py   # RF + IsolationForest + SHAP
│   └── utils/
│       └── helpers.py       # AI explanations, clinical utils
├── notebooks/
│   └── GlucoAI_Analysis.ipynb  # Full ML pipeline notebook
├── data/
│   ├── glucose_data.csv         # Sample patient data
│   └── glycemic_index.csv       # GI database (40+ foods)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── dashboard.py             # Streamlit multi-page dashboard
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick Start

### Option 1: Local Development

```bash
# 1. Clone and create virtual environment
git clone <your-repo-url>
cd glucoai
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — add GROQ_API_KEY for AI explanations (optional)

# 4. Train the ML model
python app/models/ml_pipeline.py

# 5. Start the backend API
uvicorn app.backend.api:app --reload --port 8000

# 6. Launch the dashboard (new terminal)
streamlit run dashboard.py
```

Visit:
- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

---

### Option 2: Docker

```bash
# Build and start all services
cd docker
docker-compose up --build

# Or with Groq AI:
GROQ_API_KEY=your_key docker-compose up --build
```

---

## 📡 API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/patients/` | Create patient profile |
| `GET` | `/patients/{id}` | Get patient info |
| `POST` | `/glucose/` | Log a glucose reading |
| `POST` | `/glucose/batch/` | Analyze batch readings |
| `GET` | `/glucose/{id}/history` | Fetch reading history |
| `POST` | `/predict/risk/` | Predict diabetes risk |
| `POST` | `/meals/analyze/` | Analyze meal GI |
| `POST` | `/meals/what-if/` | Compare two meals |
| `POST` | `/trends/weekly/` | Weekly glucose summary |
| `GET` | `/model/info/` | Model metadata |
| `POST` | `/model/retrain/` | Retrain model |

### Example: Risk Prediction

```bash
curl -X POST http://localhost:8000/predict/risk/ \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "bmi": 31.2,
    "glucose": 165,
    "hba1c": 7.1,
    "activity_level": 1,
    "sleep_hours": 5.5,
    "stress_level": 2,
    "family_history": 1,
    "hypertension": 1
  }'
```

Response:
```json
{
  "risk_score": 0.812,
  "risk_level": "high",
  "shap_values": { "HbA1c": 0.145, "BMI": 0.098, ... },
  "message": "High risk detected. Please consult a healthcare professional."
}
```

---

## ☁️ Deployment

### Backend → Render / Railway

```bash
# Render: Connect GitHub repo
# Set: Build command: pip install -r requirements.txt
# Set: Start command: uvicorn app.backend.api:app --host 0.0.0.0 --port $PORT

# Railway:
railway login
railway init
railway up
```

### Dashboard → Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → Set main file: `dashboard.py`
4. Add secrets in Settings: `GROQ_API_KEY = "your_key"`

---

## 🧪 Running the Notebook

```bash
cd notebooks
jupyter lab GlucoAI_Analysis.ipynb
# Run all cells — model will train and save automatically
```

---

## ⚙️ Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | No | SQLite (default) or PostgreSQL URL |
| `GROQ_API_KEY` | No | For Llama3 AI explanations |
| `SECRET_KEY` | Prod | JWT signing key |

---

## 🩺 Clinical Reference

| State | Fasting Glucose | HbA1c |
|-------|----------------|-------|
| Normal | < 100 mg/dL | < 5.7% |
| Pre-diabetes | 100–125 mg/dL | 5.7–6.4% |
| Diabetes | ≥ 126 mg/dL | ≥ 6.5% |
| Time-in-Range Target | 70–180 mg/dL | > 70% TIR |

---

## ⚠️ Disclaimer

GlucoAI is for **educational and research purposes only**. It is not a certified medical device and should not replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

## 📄 License

MIT License — see LICENSE file.
