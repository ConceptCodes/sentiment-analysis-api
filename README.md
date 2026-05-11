## Sentiment Analysis

## Commands

- `uv run sentiment-preprocess` to build the tokenizer and processed datasets
- `uv run sentiment-train` to train the classifier
- `uvicorn src.api:app --reload --host 0.0.0.0 --port 8000` to start the API
