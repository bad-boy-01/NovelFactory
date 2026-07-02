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

class ProviderCapability(BaseModel):
    modality: str = "image"
    max_resolution: tuple[int, int] = (2048, 2048)
    supports_lora: bool = False
    supports_controlnet: bool = False
    supports_img2img: bool = False
    supports_ip_adapter: bool = False
    supports_inpainting: bool = False

class PromptFingerprint(BaseModel):
    provider_name: str
    provider_version: str
    prompt_hash: str
    model_hash: str
    sampling_hash: str
    schema_hash: str
    
    @property
    def key(self) -> str:
        import hashlib
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

class ImageGenerationProvider(Protocol):
    def compile_prompt(self, visual_scene: 'VisualScene') -> 'RenderJob':
        """Compiles a fully resolved VisualScene into a provider-specific RenderJob."""
        ...
        
    def capabilities(self) -> ProviderCapability:
        ...
        
    def warmup(self) -> None:
        ...
        
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
