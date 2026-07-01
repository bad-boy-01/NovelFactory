from typing import Protocol, Dict, Any, List
from pathlib import Path

# Need to import for EvaluatorPlugin
from core.pipeline.context import PipelineContext
from core.domain.asset import Asset, EvaluationResult

class LLMProvider(Protocol):
    def generate_json(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a structured JSON response from the LLM, guaranteed to match schema."""
        ...

class ImageGeneratorProvider(Protocol):
    def get_model_name(self) -> str:
        ...
        
    def get_model_revision(self) -> str:
        ...
        
    def generate_image(self, prompt: str, negative_prompt: str, seed: int, output_path: Path) -> Path:
        """Generates an image and saves it to output_path, returning the path."""
        ...

class EvaluatorPlugin(Protocol):
    def get_name(self) -> str:
        ...
        
    def evaluate(self, asset: Asset, context: PipelineContext) -> EvaluationResult:
        ...

class VideoRendererProvider(Protocol):
    def render_video(self, image_paths: List[Path], audio_paths: List[Path], output_path: Path) -> Path:
        """Assembles images and audio into a final video."""
        ...
