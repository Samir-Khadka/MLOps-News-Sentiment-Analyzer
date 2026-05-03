from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_predict_sentiment_no_model_fallback():
    """Test prediction using mock or distilbert default if model isn't fully loaded."""
    response = client.post("/predict", json={"text": "This is a great day for Apple stock!"})
    if response.status_code == 200:
        data = response.json()
        assert "sentiment" in data
        assert "confidence" in data
    elif response.status_code == 503:
        assert response.json()["detail"] == "Model is not loaded"
