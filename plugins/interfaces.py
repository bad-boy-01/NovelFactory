from typing import Protocol, Dict, Any, List
from pathlib import Path

# Need to import for EvaluatorPlugin
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
    model_id: str = "runwayml/stable-diffusion-v1-5"
    revision: str = "main"
    dtype: torch.dtype = torch.float16
    steps: int = 25
    guidance_scale: float = 7.5
    width: int = 768
    height: int = 768
    cpu_offload: bool = True

class ImageGeneratorProvider(Protocol):
    def initialize(self) -> None:
        ...
    def load(self) -> None:
        ...
    def unload(self) -> None:
        ...
    def shutdown(self) -> None:
        ...
    def get_model_name(self) -> str:
        ...
        
    def get_model_revision(self) -> str:
        ...
        
    def generate_image(self, request: 'GenerationRequest') -> 'GeneratedImage':
        """Generates an image based on the generation request and returns a rich GeneratedImage artifact."""
        ...

class VideoRendererProvider(Protocol):
    def render_video(self, manifest: 'FrameManifest', audio_paths: List[Path], output_path: Path) -> Path:
        """Assembles frames from a manifest into a final video."""
        ...
