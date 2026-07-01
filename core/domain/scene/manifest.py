from pydantic import BaseModel
from typing import List, Optional
from .base import DomainModel

class FrameInstruction(BaseModel):
    image_asset_id: Optional[str] = None
    transition: Optional[str] = None
    duration: float = 2.0

class Shot(BaseModel):
    shot_id: str
    camera_type: str = ""
    lens: str = ""
    angle: str = ""
    distance: str = ""
    movement: str = ""
    duration: float = 2.0
    frames: List[FrameInstruction] = []

class Beat(BaseModel):
    beat_id: str
    description: str
    emotion: str
    shots: List[Shot] = []

class Scene(BaseModel):
    scene_id: str
    chapter: int
    start_offset: int
    end_offset: int
    estimated_duration: float
    characters: List[str]
    location: str
    emotion: str
    beats: List[Beat] = []

class SceneManifest(DomainModel):
    scenes: List[Scene] = []

class ShotManifest(DomainModel):
    shots: List[Shot] = []
