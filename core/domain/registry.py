from pydantic import BaseModel
from typing import Optional, Dict
from .base import DomainModel

class Asset(BaseModel):
    asset_id: str
    type: str  # "image", "audio", "video", "subtitle"
    checksum: str
    prompt_hash: Optional[str] = None
    seed: Optional[int] = None
    model_version: Optional[str] = None
    scheduler: Optional[str] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    revision: Optional[str] = None
    path: str

class AssetRegistry(DomainModel):
    assets: Dict[str, Asset] = {}
