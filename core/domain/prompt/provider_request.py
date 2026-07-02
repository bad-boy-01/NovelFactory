from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal
from core.domain.base import DomainModel

class ProviderRequest(DomainModel):
    request_type: Literal["image", "video", "audio"] = "image"
    backend: str = "diffusers"

class ImageRequest(ProviderRequest):
    request_type: Literal["image"] = "image"
    positive_prompt: str = ""
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg: float = 7.0
    seed: int = 0
    scheduler: str = "Euler a"
    loras: List[str] = Field(default_factory=list)
    controlnets: List[str] = Field(default_factory=list)
    ip_adapters: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)

class VideoRequest(ProviderRequest):
    request_type: Literal["video"] = "video"
    positive_prompt: str = ""
    negative_prompt: str = ""
    frames: int = 49
    fps: int = 24
    width: int = 1024
    height: int = 1024
    steps: int = 50
    cfg: float = 7.0
    seed: int = 0
    metadata: Dict[str, str] = Field(default_factory=dict)
