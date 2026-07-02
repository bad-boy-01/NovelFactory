from pydantic import BaseModel, Field
from typing import Optional
from core.domain.base import DomainModel

class PipelineConfig(DomainModel):
    project_id: str = "default_project"
    planning_model: str = "gpt-4o"
    diffusion_model: str = "sdxl-lightning"
    render_preset: str = "fast"
    scheduler: str = "euler_a"
    cache: bool = True
    batch_size: int = 1
    seed: int = 42
    resume: bool = True
    num_workers: int = 1
    precision: str = "fp16"
    cpu_offload: bool = False
