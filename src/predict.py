import json
import os
import torch
try:
    from .model import SentimentClassifier
    from .config import get_settings
    from .logger import get_logger
    from .schemas import PredictResponse
except ImportError:  # pragma: no cover - fallback for direct script execution
    from model import SentimentClassifier
    from config import get_settings
    from logger import get_logger
    from schemas import PredictResponse
from tokenizers import Tokenizer
from torch.nn.functional import softmax

logger = get_logger("predict")

def load_predictor():
    cfg = get_settings()
    try:
        tokenizer = Tokenizer.from_file("data/tokenizer.json")
        logger.info("Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    model = SentimentClassifier(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        pad_idx=cfg.pad_idx
    )

    def get_latest_model_path():
        model_dir = "models"
        if not os.path.exists(model_dir):
            raise FileNotFoundError("Models directory does not exist.")
        model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
        if not model_files:
            raise FileNotFoundError("No model files found in the models directory.")
        latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
        return os.path.join(model_dir, latest_model)

    try:
        latest_model_path = get_latest_model_path()
        map_location = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        model.load_state_dict(torch.load(latest_model_path, map_location=map_location))
        model.eval()
        logger.info(f"Model loaded from {latest_model_path}.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

    try:
        with open("data/processed/label_mapping.json", "r") as f:
            label_map = {int(k): v for k, v in json.load(f).items()}
        logger.info("Label mapping loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load label mapping: {e}")
        raise RuntimeError(f"Failed to load label mapping: {e}")
    
    return tokenizer, model, label_map


def predict_sentiment(text: str, tokenizer, model, label_map, max_len: int | None = None) -> PredictResponse:
    """Predict the sentiment of the given text using the loaded model."""
    try:
        if max_len is None:
            max_len = get_settings().max_len
        logger.info("Predicting sentiment for input of %s chars", len(text))
        encoding = tokenizer.encode(text)
        ids = encoding.ids[:max_len]

        if not ids:
            return PredictResponse(sentiment="Neutral", score=0.0)

        tensor = torch.tensor(ids).unsqueeze(0)
        tensor = torch.nn.functional.pad(tensor, (0, max_len - tensor.size(1)), value=0)

        with torch.no_grad():
            outputs = model(tensor)
            probs = softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)

        label = label_map.get(pred_class.item(), "Neutral")
        logger.info(f"Prediction: {label}, Score: {confidence.item()}")
        return PredictResponse(sentiment=label, score=confidence.item())
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise RuntimeError(f"Prediction failed: {e}")
