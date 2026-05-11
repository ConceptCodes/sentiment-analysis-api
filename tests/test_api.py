import pytest
from fastapi.testclient import TestClient
from src.api import app
import src.api as api_module


class _DummyEncoding:
    ids = [1, 2, 3]


class _DummyTokenizer:
    def encode(self, text):
        return _DummyEncoding()


class _DummyModel:
    def eval(self):
        return self

    def __call__(self, inputs):
        return __import__("torch").tensor([[0.1, 0.2, 0.7]])


@pytest.fixture(autouse=True)
def stub_predictor(monkeypatch):
    monkeypatch.setattr(api_module, "load_predictor", lambda: (_DummyTokenizer(), _DummyModel(), {2: "Positive"}))
    app.state.tokenizer = _DummyTokenizer()
    app.state.model = _DummyModel()
    app.state.label_map = {2: "Positive"}
    yield

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_ping(client):
    response = client.get("/ping")
    assert response.status_code == 200
    # model_loaded might be False if there's no model, but the test passes
    data = response.json()
    assert data["message"] == "pong"
    assert "model_loaded" in data

def test_predict_valid(client):
    payload = {"text": "I love this product!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "score" in data
    assert isinstance(data["sentiment"], str)
    assert isinstance(data["score"], float)

def test_predict_blank_text(client):
    payload = {"text": "   "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error

def test_predict_missing_text(client):
    payload = {}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error
