from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["message"]
    content: str = Field(min_length=1, max_length=4000)
    account_choice: int | None = Field(default=None, ge=0, le=9)
    thread_id: UUID | None = None