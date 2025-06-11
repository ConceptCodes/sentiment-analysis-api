import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "pong"}

def test_predict_valid():
    payload = {"text": "I love this product!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "score" in data
    assert isinstance(data["sentiment"], str)
    assert isinstance(data["score"], float)

def test_predict_blank_text():
    payload = {"text": "   "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error

def test_predict_missing_text():
    payload = {}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error
