import os
import requests
import json
import datetime
import pandas as pd
from prefect import task, flow, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta
import logging

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "dummy_key")
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

TICKERS = ["AAPL", "TSLA", "MSFT"]

@task(retries=3, retry_delay_seconds=60)
def extract_news():
    logger = get_run_logger()
    logger.info("Extracting news from NewsAPI...")
    
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M")
    raw_files = []
    
    for ticker in TICKERS:
        # If dummy key, mock response
        if NEWS_API_KEY == "dummy_key":
            logger.warning("Using mock data due to missing API key")
            mock_data = {
                "articles": [
                    {"title": f"Good news for {ticker}", "description": "Stock is soaring high.", "publishedAt": timestamp},
                    {"title": f"Bad news for {ticker}", "description": "Stock plummets due to earnings miss.", "publishedAt": timestamp}
                ]
            }
            file_path = os.path.join(RAW_DIR, f"{ticker}_{timestamp}.json")
            with open(file_path, "w") as f:
                json.dump(mock_data, f)
            raw_files.append(file_path)
        else:
            url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
            resp = requests.get(url)
            if resp.status_code == 200:
                data = resp.json()
                file_path = os.path.join(RAW_DIR, f"{ticker}_{timestamp}.json")
                with open(file_path, "w") as f:
                    json.dump(data, f)
                raw_files.append(file_path)
            else:
                logger.error(f"Failed to fetch news for {ticker}: {resp.status_code}")
                raise Exception(f"NewsAPI error: {resp.text}")
                
    return raw_files

@task(retries=3, retry_delay_seconds=60)
def preprocess(raw_files):
    logger = get_run_logger()
    logger.info(f"Preprocessing {len(raw_files)} files...")
    
    processed_texts = []
    for file_path in raw_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            for article in data.get("articles", []):
                text = f"{article.get('title', '')} {article.get('description', '')}".strip()
                if text:
                    # Basic cleaning
                    text = text.replace('\n', ' ').replace('\r', '')
                    processed_texts.append({"text": text, "source_file": file_path})
                    
    processed_file = os.path.join(PROCESSED_DIR, f"processed_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M')}.json")
    with open(processed_file, "w") as f:
        json.dump(processed_texts, f)
        
    return processed_texts

@task(retries=3, retry_delay_seconds=60)
def predict_sentiment(processed_texts):
    logger = get_run_logger()
    logger.info("Predicting sentiment via API...")
    
    API_URL = os.getenv("API_URL", "http://api:8000/predict")
    
    predictions = []
    for item in processed_texts:
        try:
            resp = requests.post(API_URL, json={"text": item["text"]}, timeout=10)
            if resp.status_code == 200:
                predictions.append({**item, **resp.json()})
            else:
                logger.error(f"API Error: {resp.status_code}")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            
    return predictions

@task(retries=3, retry_delay_seconds=60)
def log_to_monitoring(predictions):
    logger = get_run_logger()
    logger.info(f"Logging {len(predictions)} predictions to monitoring...")
    
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    
    # In a real scenario we compare current texts against reference.
    # Here we simulate generation of a drift report html if we had reference data
    logger.info("Drift computation simulation running...")
    drift_report_path = os.path.join(DATA_DIR, "drift_report.html")
    with open(drift_report_path, "w") as f:
        f.write("<html><body><h1>Evidently AI Mock Drift Report</h1><p>No drift detected.</p></body></html>")
        
    return drift_report_path

@flow(name="Sentinel Real-Time Sentiment Pipeline", retries=1)
def sentinel_pipeline():
    logger = get_run_logger()
    try:
        raw_files = extract_news()
        processed_texts = preprocess(raw_files)
        if processed_texts:
            predictions = predict_sentiment(processed_texts)
            log_to_monitoring(predictions)
    except Exception as e:
        logger.error(f"Pipeline failed repeatedly: {e}")
        # Send alert
        logger.error("ALERT: Pipeline execution failed!")

if __name__ == "__main__":
    sentinel_pipeline()
