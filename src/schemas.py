from pydantic import BaseModel, validator, Field
from config import Config

cfg = Config()


class PredictRequest(BaseModel):
    """Request model for sentiment prediction."""
    text: str = Field(..., min_length=1, max_length=cfg.max_len)

    @validator('text')
    def text_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError('Text must not be blank or only whitespace')
        return v
    
class PredictResponse(BaseModel):
  """Response model for sentiment prediction."""
  sentiment: str = "Neutral"
  score: float = 0.0