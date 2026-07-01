from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional

class ProvenanceGraph(BaseModel):
    generated_from: Optional[UUID] = None
    model: Optional[str] = None
    model_revision: Optional[str] = None
    prompt_hash: Optional[str] = None
    story_bible_hash: Optional[str] = None
    config_hash: Optional[str] = None
    seed: Optional[int] = None

class DomainModel(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
