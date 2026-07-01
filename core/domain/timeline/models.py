from pydantic import BaseModel
from typing import List, Optional
from .base import DomainModel

class TimelineItem(BaseModel):
    start: float
    end: float
    layer: int = 0
    asset_id: Optional[str] = None
    animation: Optional[str] = None
    subtitle_id: Optional[str] = None
    voice_id: Optional[str] = None
    music_id: Optional[str] = None
    sfx_id: Optional[str] = None
    transition: Optional[str] = None

class Timeline(DomainModel):
    duration: float = 0.0
    items: List[TimelineItem] = []
