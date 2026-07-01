from pydantic import BaseModel
from typing import Dict
from .base import DomainModel

class CharacterReference(BaseModel):
    name: str
    visual_dna: str
    outfit: str
    color_palette: str

class StoryBible(DomainModel):
    version: int = 1
    hash: str = ""
    generator_model: str = ""
    characters: Dict[str, CharacterReference] = {}
