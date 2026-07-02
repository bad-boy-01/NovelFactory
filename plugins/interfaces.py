from typing import Protocol, Dict, Any, List, TYPE_CHECKING
from pathlib import Path
from pydantic import BaseModel

if TYPE_CHECKING:
    from PIL import Image
    from core.domain.rendering.presets import RenderJob
from core.pipeline.context import PipelineContext

class LLMProvider(Protocol):
    def initialize(self) -> None:
        ...
    def load(self) -> None:
        ...
    def unload(self) -> None:
        ...
    def shutdown(self) -> None:
        ...
    def generate_json(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a structured JSON response from the LLM, guaranteed to match schema."""
        ...

from dataclasses import dataclass
import torch

@dataclass
class DiffusionConfig:
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    cache_dir: str = "workspace/models"
    revision: str = "main"
    dtype: torch.dtype = torch.float16
    cpu_offload: bool = True

class ProviderHealth(BaseModel):
    loaded: bool
    device: str
    model: str
    dtype: str
    vram_allocated_gb: float

class ImageGenerationProvider(Protocol):
    def load(self) -> None:
        ...
        
    def generate(self, job: 'RenderJob', callback=None) -> Image.Image:
        ...
        
    def health_check(self) -> ProviderHealth:
        ...
        
    def unload(self) -> None:
        ...

class VideoRendererProvider(Protocol):
    def render_video(self, manifest: 'FrameManifest', audio_paths: List[Path], output_path: Path) -> Path:
        """Assembles frames from a manifest into a final video."""
        ...
