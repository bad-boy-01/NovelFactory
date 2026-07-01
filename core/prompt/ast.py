from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptAST:
    character: str
    outfit: str
    scene: str
    camera: str
    lighting: str
    style: str
    negative: Optional[str] = None
