"""Chat message models."""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class ChatMessage(BaseModel):
    """Chat message model for database storage."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    thinking: Optional[bool] = False
    model: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Hello, Isabella!",
                "timestamp": "2024-01-01T00:00:00",
                "thinking": False,
                "model": "LongCat-Flash-Chat"
            }
        }
