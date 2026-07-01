from pydantic import BaseModel
from typing import Dict, List, Optional
from core.domain.base import DomainModel

class AssetDependencies(BaseModel):
    prompt_manifest_hash: str = ""
    character_render_state_hash: str = ""
    model_version: str = "sdxl_1.0"
    loras: List[str] = []

class Asset(BaseModel):
    asset_id: str
    type: str # "image", "audio", "video"
    path: str
    checksum: str
    prompt_hash: str
    seed: int
    dependencies: Optional[AssetDependencies] = None
    created_at: float = 0.0

class AssetRegistry(DomainModel):
    """
    Permanent, reproducible, tracked canonical assets.
    """
    assets: Dict[str, Asset] = {}
