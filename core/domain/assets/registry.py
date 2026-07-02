from pydantic import BaseModel
from typing import Dict, List, Optional
from core.domain.base import DomainModel

from enum import Enum
import hashlib
from pathlib import Path

class AssetStatus(Enum):
    VALID = "valid"
    MISSING_FILE = "missing_file"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    OUTDATED = "outdated"
    STALE_METADATA = "stale_metadata"
    ORPHANED = "orphaned"
    
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

    def get_asset_status(self, asset_id: str, expected_prompt_hash: Optional[str] = None) -> AssetStatus:
        if asset_id not in self.assets:
            return AssetStatus.ORPHANED
            
        asset = self.assets[asset_id]
        asset_path = Path(asset.path)
        
        if not asset_path.exists():
            return AssetStatus.MISSING_FILE
            
        if expected_prompt_hash and asset.prompt_hash != expected_prompt_hash:
            return AssetStatus.STALE_METADATA
            
        # Optional: implement checksum verification here if desired
        
        return AssetStatus.VALID
        
    def register_asset(self, asset: Asset):
        self.assets[asset.asset_id] = asset

    def load(self, workspace_manager) -> None:
        data = workspace_manager.load_json("AssetRegistry.json")
        if data:
            self.assets = {k: Asset(**v) for k, v in data.get("assets", {}).items()}
            
    def save(self, workspace_manager) -> None:
        workspace_manager.save_json("AssetRegistry.json", self.model_dump())
