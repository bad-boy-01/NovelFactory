from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import datetime
from pathlib import Path


@dataclass(frozen=True)
class ProvenanceRecord:
    """Immutable record of how an asset was generated."""
    model_id: str
    revision: str
    prompt_hash: str
    seed: int
    scheduler: str
    guidance_scale: float
    inference_steps: int
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    diffusers_version: Optional[str] = None
    torch_version: Optional[str] = None


@dataclass
class GenerationRequest:
    """Immutable request object encapsulating all generation parameters."""
    compiled_prompt: str
    negative_prompt: str
    seed: int
    prompt_hash: str
    model_id: str
    output_path: Path
    width: int
    height: int
    steps: int
    guidance_scale: float


@dataclass
class GeneratedImage:
    """The central artifact for generated visual assets."""
    image_path: Path
    width: int
    height: int
    seed: int
    prompt_hash: str
    model_id: str
    cache_hit: bool
    provenance: Optional[ProvenanceRecord] = None


@dataclass(frozen=True)
class FrameManifestEntry:
    """A single frame mapped to its chronological position and source."""
    frame_index: int
    beat_id: str
    image_path: Path
    prompt_hash: str
    asset_id: str


@dataclass(frozen=True)
class FrameManifest:
    """The complete manifest consumed by the Renderer."""
    frames: List[FrameManifestEntry]


@dataclass(frozen=True)
class ExecutionNode:
    """The canonical runtime object grouping an artifact with its provenance and generation request."""
    artifact: Any
    request: Optional[GenerationRequest] = None
    provenance: Optional[ProvenanceRecord] = None
    cache_key: str = ""
    stage_name: str = ""
    execution_time: float = 0.0
    retry_count: int = 0
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    contract_results: List[Any] = field(default_factory=list)  # List[ContractResult]
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())


class Asset:
    """Base asset tracking class for the context."""
    def __init__(self, asset_id: str):
        self.asset_id = asset_id
        self.generated_image: Optional[GeneratedImage] = None
        
    def apply(self, context: Any) -> None:
        """Commits this asset to the pipeline context."""
        if not hasattr(context, 'assets'):
            context.assets = []
        context.assets.append(self)
