import json
import argparse
import pandas as pd
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, normalizers
from tokenizers.trainers import WordLevelTrainer
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

try:
    from .config import Config
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import Config

cfg = Config()
MAX_LEN = cfg.max_len
TOKENIZER_PATH = "data/tokenizer.json"

TRAIN_PATH = "data/cleaned/twitter_training.csv"
VAL_PATH = "data/cleaned/twitter_validation.csv"

SAVE_DIR = Path("data/processed")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def tokenize_and_pad(texts, tokenizer):
    token_ids = []
    for text in texts:
        encoding = tokenizer.encode(text)
        ids = encoding.ids[:MAX_LEN]
        token_ids.append(torch.tensor(ids))
    return pad_sequence(token_ids, batch_first=True, padding_value=0)

def process_and_save(df, name, tokenizer, output_dir: Path):
    print(f"Processing {name}...")
    texts = df.columns[-1]
    labels = df.columns[-2]

    df[texts] = df[texts].astype(str)
    df[labels] = df[labels].astype(str)

    input_ids = tokenize_and_pad(df[texts], tokenizer)

    label_values, _ = pd.factorize(df[labels])

    labels_tensor = torch.tensor(label_values)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save((input_ids, labels_tensor), output_dir / f"{name}.pt")
    print(f"✅ Saved {name} dataset to {output_dir}/{name}.pt")

def train_tokenizer(df: pd.DataFrame, vocab_size: int = 10000, output_path: str = "data/tokenizer.json"):
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()

    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    tokenizer.train_from_iterator(df.iloc[:, -1].astype(str).tolist(), trainer=trainer)

    tokenizer.save(output_path)
    print(f"Tokenizer trained and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Clean data, train tokenizer, and build processed datasets.")
    parser.add_argument("--train-path", default=TRAIN_PATH, help="Path to the raw training CSV.")
    parser.add_argument("--val-path", default=VAL_PATH, help="Path to the raw validation CSV.")
    parser.add_argument("--tokenizer-path", default=TOKENIZER_PATH, help="Where to save the tokenizer JSON.")
    parser.add_argument("--processed-dir", default=str(SAVE_DIR), help="Directory for processed tensor files.")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    train_df = pd.DataFrame(train_df)

    val_df = pd.read_csv(args.val_path)
    val_df = pd.DataFrame(val_df)

    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    train_tokenizer(combined_df, vocab_size=cfg.vocab_size, output_path=args.tokenizer_path)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    process_and_save(train_df, "train", tokenizer, processed_dir)
    process_and_save(val_df, "val", tokenizer, processed_dir)

    label_mapping = {i: label for i, label in enumerate(train_df[train_df.columns[-2]].unique())}
    with open(processed_dir / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f)
    print(f"✅ Saved label mapping to {processed_dir}/label_mapping.json")


if __name__ == "__main__":
    main()
