from core.pipeline.stage import PipelineStage, StageResult
from core.domain.asset import ExecutionNode
from core.domain.scene import SceneManifest, Scene, Beat
import json
import logging
import hashlib

logger = logging.getLogger(__name__)

class SceneSplitterStage(PipelineStage):
    def __init__(self, llm_provider=None):
        self.llm = llm_provider

    def get_providers(self) -> list:
        return [self.llm] if self.llm else []

    def execute(self, context) -> StageResult:
        if not self.llm:
            from plugins.local_llm import LocalLLMProvider
            self.llm = LocalLLMProvider()
            
        raw_text = context.project_manifest.source_text
        
        # Safety limit for Milestone 1/2 VRAM constraints
        if len(raw_text) > 10000:
            raw_text = raw_text[:10000] + "\n...[TRUNCATED]"
            
        # Deterministic hashing of the input to ensure stability
        text_hash = hashlib.sha256(raw_text.encode('utf-8')).hexdigest()[:8]
            
        schema = {
            "scenes": [
                {
                    "scene_id": "string",
                    "chapter": 1,
                    "estimated_duration": 15.5,
                    "characters": ["string"],
                    "location": "string",
                    "emotion": "string",
                    "beats": [
                        {
                            "beat_id": "string",
                            "description": "string",
                            "emotion": "string"
                        }
                    ]
                }
            ]
        }
        
        prompt = f"""
You are a master cinematic story planner. Your task is to split the following text into distinct narrative scenes.
CRITICAL RULES:
- Never summarize. Expand the story to capture every detail.
- Preserve every event, dialogue, and emotional beat.
- Split the text into logical scene boundaries (e.g., location changes, time skips, major shifts in action).
- Assign a unique scene_id starting with 'scene_' and a beat_id starting with 'beat_'.
- scene_id must be stable and deterministic. Example: scene_{text_hash}_001

Text to process:
{raw_text}
"""
        
        result_dict = self.llm.generate_json(prompt, schema)
        
        scenes = []
        for s in result_dict.get("scenes", []):
            beats = [Beat(**b) for b in s.get("beats", [])]
            scene = Scene(
                scene_id=s["scene_id"],
                chapter=s.get("chapter", 1),
                start_offset=0, # Simplified for Milestone 2 testing
                end_offset=0,   # Simplified for Milestone 2 testing
                estimated_duration=s.get("estimated_duration", 10.0),
                characters=s.get("characters", []),
                location=s.get("location", ""),
                emotion=s.get("emotion", ""),
                beats=beats
            )
            scenes.append(scene)
            
        manifest = SceneManifest(
            scenes=scenes,
            generator="SceneSplitterStage",
            generator_version="0.1.0",
            source_hash=text_hash,
            schema_version="2.0"
        )
        
        node = ExecutionNode(artifact=manifest, stage_name="SceneSplitterStage")
        
        return StageResult(
            artifact=manifest,
            execution_node=node,
            metrics={"scenes_extracted": len(scenes)},
            metadata={}
        )
