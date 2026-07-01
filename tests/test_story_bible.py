import pytest
from typing import Dict, Any
from core.pipeline.context import PipelineContext
from core.domain.project import ProjectManifest
from ai.reasoning.story_bible import StoryBibleGeneratorStage

class MockLLMProvider:
    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        return "text"
        
    def generate_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        return {
            "characters": [
                {
                    "name": "Alice",
                    "visual_dna": "tall, blonde hair",
                    "outfit": "blue school uniform",
                    "color_palette": "blue, white"
                }
            ]
        }

class FailingLLMProvider:
    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        raise ValueError("API Error")
        
    def generate_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        raise ValueError("API Error")

def test_story_bible_generator():
    manifest = ProjectManifest(project_name="Test", dataset_id="Dataset")
    ctx = PipelineContext(project_manifest=manifest)
    
    llm = MockLLMProvider()
    stage = StoryBibleGeneratorStage(llm=llm)
    
    assert stage.get_name() == "Story Bible Generator"
    
    ctx = stage.execute(ctx)
    
    assert ctx.story_bible is not None
    assert "alice" in ctx.story_bible.characters
    assert ctx.story_bible.characters["alice"].outfit == "blue school uniform"

def test_story_bible_generator_fallback():
    manifest = ProjectManifest(project_name="Test", dataset_id="Dataset")
    ctx = PipelineContext(project_manifest=manifest)
    
    llm = FailingLLMProvider()
    stage = StoryBibleGeneratorStage(llm=llm)
    
    ctx = stage.execute(ctx)
    
    assert ctx.story_bible is not None
    assert ctx.story_bible.generator_model == "error_fallback"
    assert len(ctx.story_bible.characters) == 0

def test_story_bible_generator_skips_if_exists():
    manifest = ProjectManifest(project_name="Test", dataset_id="Dataset")
    ctx = PipelineContext(project_manifest=manifest)
    
    llm = MockLLMProvider()
    stage = StoryBibleGeneratorStage(llm=llm)
    
    # First execution populates it
    ctx = stage.execute(ctx)
    
    # Modify it manually
    ctx.story_bible.version = 999
    
    # Second execution should skip
    ctx = stage.execute(ctx)
    
    assert ctx.story_bible.version == 999
