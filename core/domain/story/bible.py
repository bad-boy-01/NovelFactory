from pydantic import BaseModel
from typing import Dict
from .base import DomainModel

class CharacterReference(BaseModel):
    name: str
    visual_dna: str
    hair: str = ""
    eyes: str = ""
    face: str = ""
    body: str = ""

class CharacterState(BaseModel):
    outfit: str = ""
    condition: str = ""
    emotion: str = ""
    accessories: list[str] = []

class CharacterSnapshot(BaseModel):
    reference: CharacterReference
    state: CharacterState
    scene_id: str = ""

class StoryBible(DomainModel):
    version: int = 2
    hash: str = ""
    generator_model: str = ""
    characters: Dict[str, CharacterReference] = {}
