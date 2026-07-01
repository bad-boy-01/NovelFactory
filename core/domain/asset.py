from enum import Enum
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
from .base import DomainModel, ProvenanceGraph

class AssetType(str, Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    JSON = "json"

class EvaluationResult(BaseModel):
    score: float
    reason: str
    retry_needed: bool

class Asset(DomainModel):
    asset_type: AssetType
    file_path: Path
    hash_sha256: str
    provenance: ProvenanceGraph
    evaluation: Optional[EvaluationResult] = None
