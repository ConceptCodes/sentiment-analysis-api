from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Config(BaseSettings):
    """
    Configuration settings for the application, loaded from environment variables if available.
    """

    max_len: int = Field(
        default=512,
        description="Maximum length of the input text sequences.",
        alias="MAX_LEN",
    )

    batch_size: int = Field(
        default=32,
        description="Batch size for training.",
        alias="BATCH_SIZE",
    )
    epochs: int = Field(
        default=5,
        description="Number of training epochs.",
        alias="EPOCHS",
    )
    embed_dim: int = Field(
        default=128,
        description="Embedding dimension size.",
        alias="EMBED_DIM",
    )
    hidden_dim: int = Field(
        default=256,
        description="Hidden layer dimension size.",
        alias="HIDDEN_DIM",
    )
    vocab_size: int = Field(
        default=30522,
        description="Size of the vocabulary for the tokenizer.",
        alias="VOCAB_SIZE",
    )
    output_dim: int = Field(
        default=3,
        description="Number of output classes.",
        alias="OUTPUT_DIM",
    )
    pad_idx: int = Field(
        default=0,
        description="Padding index for sequences.",
        alias="PAD_IDX",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

@lru_cache
def get_settings() -> Config:
    return Config()
