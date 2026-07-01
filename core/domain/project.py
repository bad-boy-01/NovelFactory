from pydantic import BaseModel
from .base import DomainModel

class QualityPreset(BaseModel):
    resolution: tuple[int, int] = (1080, 1920)
    inference_steps: int = 20
    cfg_scale: float = 7.0

class ProjectManifest(DomainModel):
    project_name: str
    dataset_id: str
    source_text: str = ""
    quality_preset: QualityPreset = QualityPreset()
