FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Start prefect worker and scheduler
# For demonstration we can just run the pipeline using a simple loop or prefect schedule
CMD ["python", "sentinel_pipeline.py"]
