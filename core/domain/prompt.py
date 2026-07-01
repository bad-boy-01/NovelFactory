from pydantic import BaseModel
from typing import List, Optional
from .base import DomainModel

class CameraAST(BaseModel):
    type: str = ""
    lens: str = ""
    angle: str = ""
    distance: str = ""
    movement: str = ""

class PromptAST(BaseModel):
    subject: str = ""
    characters: List[str] = []
    environment: str = ""
    camera: CameraAST = CameraAST()
    lighting: str = ""
    composition: str = ""
    style: str = ""
    negative: str = ""
    technical: str = ""

class PromptManifestEntry(BaseModel):
    prompt_id: str
    scene_id: str
    shot_id: str
    ast: PromptAST
    seed: int
    model_target: str = "sd1.5"
    width: int = 768
    height: int = 768
    steps: int = 25
    cfg: float = 7.0
    
class PromptManifest(DomainModel):
    prompts: List[PromptManifestEntry] = []
