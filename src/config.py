from pydantic import BaseSettings, Field


class Config(BaseSettings):
    """
    Configuration settings for the application, loaded from environment variables if available.
    """

    max_len: int = Field(
        default=512,
        description="Maximum length of the input text sequences.",
        env="MAX_LEN",
    )

    batch_size: int = Field(
        default=32,
        description="Batch size for training.",
        env="BATCH_SIZE",
    )
    epochs: int = Field(
        default=5,
        description="Number of training epochs.",
        env="EPOCHS",
    )
    embed_dim: int = Field(
        default=128,
        description="Embedding dimension size.",
        env="EMBED_DIM",
    )
    hidden_dim: int = Field(
        default=256,
        description="Hidden layer dimension size.",
        env="HIDDEN_DIM",
    )
    vocab_size: int = Field(
        default=30522,
        description="Size of the vocabulary for the tokenizer.",
        env="VOCAB_SIZE",
    )
    output_dim: int = Field(
        default=3,
        description="Number of output classes.",
        env="OUTPUT_DIM",
    )
    pad_idx: int = Field(
        default=0,
        description="Padding index for sequences.",
        env="PAD_IDX",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
