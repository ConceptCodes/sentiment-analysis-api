from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request

try:
    from .predict import load_predictor, predict_sentiment
    from .logger import get_logger
    from .schemas import PredictRequest, PredictResponse
except ImportError:  # pragma: no cover - fallback for direct script execution
    from predict import load_predictor, predict_sentiment
    from logger import get_logger
    from schemas import PredictRequest, PredictResponse

logger = get_logger("api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and configurations on startup
    try:
        tokenizer, model, label_map = load_predictor()
        app.state.tokenizer = tokenizer
        app.state.model = model
        app.state.label_map = label_map
        logger.info("Predictor loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load predictor: {e}")
        # Depending on requirements, we can let it fail or start without it
        # and handle errors in the predict endpoint.
        app.state.tokenizer = None
        app.state.model = None
        app.state.label_map = None
    yield
    # Cleanup on shutdown
    app.state.tokenizer = None
    app.state.model = None
    app.state.label_map = None


app = FastAPI(lifespan=lifespan)

@app.get("/ping")
def ping() -> dict:
    """Health check endpoint."""
    return {"message": "pong", "model_loaded": getattr(app.state, "model", None) is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    """Predict the sentiment of the provided text."""
    if getattr(request.app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")

    try:
        result = predict_sentiment(
            text=req.text,
            tokenizer=request.app.state.tokenizer,
            model=request.app.state.model,
            label_map=request.app.state.label_map
        )
        logger.info("Request received: %s chars", len(req.text))
        logger.info(f"Predicted sentiment: {result.sentiment}, Score: {result.score}")
        return result
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
