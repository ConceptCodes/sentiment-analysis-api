import json
import os
import torch
from model import SentimentClassifier
from tokenizers import Tokenizer
from torch.nn.functional import softmax
import logging

from config import Config
from logger import get_logger
from schemas import PredictResponse

logging.basicConfig(level=logging.INFO)

cfg = Config()
logger = get_logger("predict")

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



def predict_sentiment(text: str, max_len: int = 128) -> PredictResponse:
    """Predict the sentiment of the given text using the loaded model.

    Args:
        text (str): The input text to analyze.
        max_len (int): Maximum sequence length for tokenization.

    Returns:
        PredictResponse: The prediction result with sentiment and score.

    Raises:
        RuntimeError: If prediction fails for any reason.
    """
    try:
        logger.info(f"Predicting sentiment for text: {text}")
        encoding = tokenizer.encode(text)
        ids = encoding.ids[:max_len]
        tensor = torch.tensor(ids).unsqueeze(0)
        tensor = torch.nn.functional.pad(tensor, (0, max_len - tensor.size(1)), value=0)

        with torch.no_grad():
            outputs = model(tensor)
            probs = softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)

        label = label_map[pred_class.item()]
        logger.info(f"Prediction: {label}, Score: {confidence.item()}")
        return PredictResponse(sentiment=label, score=confidence.item())
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise RuntimeError(f"Prediction failed: {e}")
