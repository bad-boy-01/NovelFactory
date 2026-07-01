from typing import Optional, List, Dict
from pydantic import BaseModel
from .base import DomainModel

class DeclarativePrompt(DomainModel):
    """
    Abstract Syntax Tree representation of a prompt before compilation.
    """
    characters: Dict[str, str] = {}  # e.g., {"Alice": "School uniform"}
    expression: str = ""
    camera: str = ""
    lighting: str = ""
    environment: str = ""
    style: str = ""
    custom_additions: List[str] = []
    negative_constraints: List[str] = []
    
    compiled_text: str = "" # The final model-ready string
