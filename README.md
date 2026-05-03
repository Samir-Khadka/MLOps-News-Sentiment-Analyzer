# Real-Time News Sentiment Analyzer for Stock Movement

This is an end-to-end MLOps system that collects live financial news, predicts sentiment using a fine-tuned DistilBERT model, serves predictions via a REST API, and monitors for data drift & performance degradation.

## Features

- **Data Ingestion**: Fetches news from NewsAPI hourly for defined tickers.
- **Model Inference**: FastAPI serving a DistilBERT sequence classification model.
- **Orchestration**: Prefect pipeline to extract, preprocess, predict, and log.
- **Monitoring**: Real-time Plotly Dash dashboard and Evidently AI drift reports.
- **CI/CD**: GitHub Actions for linting, testing, and building Docker images.

## Quickstart

1. Ensure you have Docker and Docker Compose installed.
2. (Optional) Set your `NEWS_API_KEY` in `docker-compose.yml` or as an environment variable. If none is provided, it uses mock data.
3. Run the following command:

```bash
docker-compose up --build
```

### Accessing Services

- **FastAPI Endpoint**: `http://localhost:8000/docs`
- **Monitoring Dashboard**: `http://localhost:8050`
- **Prometheus Metrics**: `http://localhost:9090`

## Architecture

- `api/`: FastAPI backend containing the inference logic.
- `monitoring/`: Dash application for real-time visualization.
- `orchestration/`: Prefect tasks and flows.
- `src/`: Shared data preprocessing and model training scripts.

## DVC Setup
This project uses DVC to manage datasets (`data/raw`, `data/processed`) and model weights (`models/`). 
Run `dvc pull` if connected to a remote storage to retrieve the latest weights.
