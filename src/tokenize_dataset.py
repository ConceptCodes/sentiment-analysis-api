import json
import pandas as pd
import torch
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

from config import Config

cfg = Config()
MAX_LEN = cfg.max_len
TOKENIZER_PATH = "data/tokenizer.json"

TRAIN_PATH = "data/cleaned/twitter_training.csv"
VAL_PATH = "data/cleaned/twitter_validation.csv"

SAVE_DIR = Path("data/processed")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

def tokenize_and_pad(texts):
    token_ids = []
    for text in texts:
        encoding = tokenizer.encode(text)
        ids = encoding.ids[:MAX_LEN]
        token_ids.append(torch.tensor(ids))
    return pad_sequence(token_ids, batch_first=True, padding_value=0)

def process_and_save(df, name):
    print(f"Processing {name}...")
    texts = df.columns[-1]
    labels = df.columns[-2]

    df[texts] = df[texts].astype(str)
    df[labels] = df[labels].astype(str)

    input_ids = tokenize_and_pad(df[texts])

    label_values, _ = pd.factorize(df[labels])

    labels_tensor = torch.tensor(label_values)

    torch.save((input_ids, labels_tensor), SAVE_DIR / f"{name}.pt")
    print(f"✅ Saved {name} dataset to {SAVE_DIR}/{name}.pt")

train_df = pd.read_csv(TRAIN_PATH)
train_df = pd.DataFrame(train_df)

val_df = pd.read_csv(VAL_PATH)
val_df = pd.DataFrame(val_df)

process_and_save(train_df, "train")
process_and_save(val_df, "val")

label_mapping = {i: label for i, label in enumerate(train_df[train_df.columns[-2]].unique())}
with open(SAVE_DIR / "label_mapping.json", "w") as f:
    json.dump(label_mapping, f)
print(f"✅ Saved label mapping to {SAVE_DIR}/label_mapping.json")
