from pydantic import BaseModel, field_validator, Field

class PredictRequest(BaseModel):
    """Request model for sentiment prediction."""
    text: str = Field(..., min_length=1)

    @field_validator('text')
    @classmethod
    def text_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError('Text must not be blank or only whitespace')
        return v

class PredictResponse(BaseModel):
  """Response model for sentiment prediction."""
  sentiment: str = "Neutral"
  score: float = 0.0
