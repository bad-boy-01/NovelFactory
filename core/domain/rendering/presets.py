from pydantic import BaseModel
from typing import Any, Callable, Optional
from diffusers import EulerDiscreteScheduler

class RenderPreset(BaseModel):
    name: str = "lightning"
    width: int = 1024
    height: int = 1024
    steps: int = 4
    cfg: float = 0.0
    scheduler_class: Any = EulerDiscreteScheduler
    negative_prompt: str = ""
    
class RenderJob(BaseModel):
    prompt: str
    negative_prompt: str
    seed: int
    preset: RenderPreset
    # Future extensibility: lora, controlnet, etc.
