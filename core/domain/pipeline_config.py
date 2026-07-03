from pydantic import BaseModel, Field
from typing import Optional
from core.domain.base import DomainModel

class PipelineConfig(DomainModel):
    project_id: str = "default_project"
    planning_model: str = "gpt-4o"
    llm_model: str = "Qwen/Qwen1.5-4B-Chat"
    diffusion_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    render_preset: str = "fast"
    scheduler: str = "euler_a"
    cache: bool = True
    batch_size: int = 1
    seed: int = 42
    resume: bool = True
    num_workers: int = 1
    precision: str = "fp16"
    dtype: str = "float16"
    cache_dir: str = "workspace/models"
    cpu_offload: bool = True
