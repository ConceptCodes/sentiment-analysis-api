from fastapi import FastAPI, HTTPException

from predict import predict_sentiment
from config import Config
from logger import get_logger
from schemas import PredictRequest, PredictResponse

cfg = Config()
logger = get_logger("api")


app = FastAPI()

@app.get("/ping")
def ping() -> dict:
    """Health check endpoint."""
    return {"message": "pong"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict the sentiment of the provided text."""
    try:
        result = predict_sentiment(req.text)
        logger.info(f"Request text: {req.text}")
        logger.info(f"Predicted sentiment: {result.sentiment}, Score: {result.score}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
