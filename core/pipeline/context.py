from typing import Dict, Optional
from uuid import UUID
from pydantic import BaseModel, Field

from core.domain.project import ProjectManifest
from core.domain.bible import StoryBible
from core.domain.asset import Asset
from core.domain.prompt import DeclarativePrompt
from core.domain.story import Chapter, Scene

class PipelineContext(BaseModel):
    """
    The central state object that flows through the execution pipeline.
    Stages receive this read-only context. It is never mutated.
    Instead, a ContextReducer generates a new instance.
    """
    model_config = {"frozen": True}
    
    project_manifest: ProjectManifest
    story_bible: Optional[StoryBible] = None
    
    # Active scope for the current execution batch
    current_chapter: Optional[Chapter] = None
    current_scene: Optional[Scene] = None
    
    # Tracked entities generated during execution
    assets: Dict[UUID, Asset] = Field(default_factory=dict)
    prompts: Dict[UUID, DeclarativePrompt] = Field(default_factory=dict)
    
    # Ephemeral state for inter-stage communication of non-domain data
    state: dict = Field(default_factory=dict)
