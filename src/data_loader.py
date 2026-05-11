import os
import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, normalizers
from tokenizers.trainers import WordLevelTrainer
try:
    from .config import Config
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import Config

cfg = Config()

if not os.path.exists("data/cleaned"):
    os.makedirs("data/cleaned")
if not os.path.exists("data/processed"):
    os.makedirs("data/processed")


def clean_data(file_path: str) -> pd.DataFrame:
  dataset = pd.read_csv(file_path)
  df = pd.DataFrame(dataset)

  text_col = df.columns[-1]
  label_col = df.columns[-2]

  df[text_col] = df[text_col].astype(str)
  df[label_col] = df[label_col].astype(str)

  clean_df = df[
      (df[text_col].str.strip() != "") & (df[label_col].str.strip() != "")
  ].copy()

  clean_df[text_col] = clean_df[text_col].str.strip()
  clean_df[label_col] = clean_df[label_col].str.strip()

  clean_df = clean_df.drop_duplicates(subset=[text_col])

  clean_df = clean_df.reset_index(drop=True)
  output_file_path = file_path.replace("raw", "cleaned")
  
  clean_df.to_csv(output_file_path, index=False)
  print(f"Cleaned data saved to {output_file_path}")

  return clean_df


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


if __name__ == "__main__":
    test_file_path = "data/raw/twitter_training.csv"
    validation_file_path = "data/raw/twitter_validation.csv"

    cleaned_train_df = clean_data(test_file_path)
    cleaned_validation_df = clean_data(validation_file_path)

    combined_df = pd.concat([cleaned_train_df, cleaned_validation_df], ignore_index=True)

    train_tokenizer(combined_df, vocab_size=cfg.vocab_size, output_path="data/tokenizer.json")
