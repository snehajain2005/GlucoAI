FROM python:3.11-slim

# Set working directory
WORKDIR /glucoai

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p app/models data

# Pre-train model during build (optional — comment out for faster builds)
RUN python -c "from app.models.ml_pipeline import DiabetesRiskModel, generate_synthetic_dataset; \
    m = DiabetesRiskModel(); \
    df = generate_synthetic_dataset(1000); \
    m.train(df); \
    m.save(); \
    print('Model pre-trained')" || echo "Model will train on first request"

# Expose ports
EXPOSE 8000 8501

# Default: FastAPI
CMD ["uvicorn", "app.backend.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
