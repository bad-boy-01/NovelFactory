from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.story.bible import StoryBible, CharacterVisualProfile, Location, Appearance
from core.pipeline.context import PipelineContext
from typing import Any
import hashlib

class StoryBibleGeneratorStage(CompilerStage):
    def __init__(self, llm_provider):
        self.llm = llm_provider

    def get_name(self) -> str:
        return "StoryBibleGeneratorStage"

    def get_providers(self) -> list:
        return [self.llm] if self.llm else []
        
    def inputs(self, context: PipelineContext) -> list[Any]:
        return [context.project_manifest]
        
    def outputs(self) -> list[str]:
        return ["story_bible"]
        
    def generator_signature(self) -> str:
        return f"{self.get_name()}_{type(self.llm).__name__ if self.llm else 'default'}_v2.0"

    def execute(self, context: PipelineContext) -> StageResult:
        schema = {
            "characters": [
                {
                    "id": "string",
                    "name": "string",
                    "appearance": {
                        "hair": "string",
                        "eyes": "string",
                        "face": "string",
                        "age": "string",
                        "body": "string",
                        "clothing": "string",
                        "color_palette": ["string"],
                        "signature": ["string"]
                    }
                }
            ],
            "locations": [
                {
                    "id": "string",
                    "name": "string",
                    "appearance": "string",
                    "architecture": "string",
                    "weather_defaults": "string",
                    "time_defaults": "string",
                    "lighting_presets": "string"
                }
            ]
        }
        
        raw_text = context.project_manifest.source_text
        if len(raw_text) > 10000:
            raw_text = raw_text[:10000] + "\n...[TRUNCATED]"
            
        prompt = f"""
You are an expert cinematic production designer. Extract the following from the story:
1. Detailed Character Visual Profiles (physical traits, signature accessories, clothing).
2. Locations (architecture, weather, lighting defaults).

Story:
{raw_text}
"""
        
        result_dict = self.llm.generate_json(prompt, schema)
        
        chars = {}
        for c in result_dict.get("characters", []):
            try:
                char_id = c.get("id", c.get("name", "").lower().replace(" ", "_"))
                app_data = c.get("appearance", {})
                chars[char_id] = CharacterVisualProfile(
                    id=char_id,
                    name=c.get("name", "Unknown"),
                    appearance=Appearance(**app_data)
                )
            except Exception:
                pass
                
        locs = {}
        for l in result_dict.get("locations", []):
            try:
                loc_id = l.get("id", l.get("name", "").lower().replace(" ", "_"))
                locs[loc_id] = Location(**l)
            except Exception:
                pass
        
        bible = StoryBible(
            characters=chars,
            locations=locs
        )
        
        # Enveloping is now handled by the executor, we just return the raw DomainModel
        node = ExecutionNode(artifact=bible, stage_name="StoryBibleGeneratorStage")
        
        # NOTE: We temporarily save the non-enveloped file here if legacy code relies on it, 
        # but the executor handles writing the true `workspace/manifests/story_bible.json`
        
        return StageResult(
            artifact=bible,
            execution_node=node,
            metrics={"characters": len(chars), "locations": len(locs)},
            metadata={}
        )
