from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
from prometheus_fastapi_instrumentator import Instrumentator
import datetime
import json

app = FastAPI(title="Real-Time News Sentiment Analyzer")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiment: str
    confidence: float

# Initialize model placeholder
sentiment_pipeline = None
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models/distilbert-finance")
MODEL_VERSION = "1.0" # Could be dynamic based on DVC
LOG_DIR = os.getenv("LOG_DIR", "/app/data/processed/logs")

os.makedirs(LOG_DIR, exist_ok=True)

@app.on_event("startup")
async def load_model():
    global sentiment_pipeline
    try:
        # Load custom fine-tuned model if exists, else fallback to standard distilbert
        if os.path.exists(MODEL_DIR):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
        else:
            print(f"Model not found at {MODEL_DIR}, using default distilbert for demonstration")
            sentiment_pipeline = pipeline("text-classification", model="distilbert-base-uncased")
    except Exception as e:
        print(f"Error loading model: {e}")

# Instrument FastAPI for Prometheus
Instrumentator().instrument(app).expose(app)

def log_prediction(text: str, sentiment: str, confidence: float):
    log_file = os.path.join(LOG_DIR, "predictions.jsonl")
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "input_text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "model_version": MODEL_VERSION
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

@app.post("/predict", response_model=PredictResponse)
async def predict_sentiment(request: PredictRequest):
    if sentiment_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    try:
        # Truncate text to 512 tokens implicitly handled by tokenizer args if configured, 
        # but let's assume pipeline handles basic truncation or we pass truncation=True
        results = sentiment_pipeline(request.text, truncation=True, max_length=512)
        
        # results is typically [[{'label': 'LABEL_0', 'score': 0.1}, ...]] if return_all_scores=True
        # For simple pipeline it might just be [{'label': 'POSITIVE', 'score': 0.9}]
        # Let's handle generic case
        
        if isinstance(results[0], list):
            best_result = max(results[0], key=lambda x: x['score'])
        else:
            best_result = results[0]

        label = best_result['label'].lower()
        score = float(best_result['score'])
        
        # Map labels if needed (e.g., LABEL_0 to negative, etc. depending on fine-tuning)
        if "label_0" in label or "negative" in label:
            sentiment = "negative"
        elif "label_2" in label or "positive" in label:
            sentiment = "positive"
        else:
            sentiment = "neutral"

        log_prediction(request.text, sentiment, score)
        return PredictResponse(sentiment=sentiment, confidence=score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": sentiment_pipeline is not None}
