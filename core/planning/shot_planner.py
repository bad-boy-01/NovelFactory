from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.scene.manifest import SceneManifest, ShotManifest, Shot, FrameInstruction
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

class ShotPlannerStage(PipelineStage):
    def __init__(self, llm_provider=None):
        self.llm = llm_provider

    def get_name(self) -> str:
        return "ShotPlannerStage"

    def fingerprint(self, context) -> str:
        # Generate a fingerprint based on the class name and provider type
        base = f"{self.get_name()}_{type(self.llm).__name__ if self.llm else 'default'}"
        return hashlib.sha256(base.encode('utf-8')).hexdigest()

    def get_providers(self) -> list:
        return [self.llm] if self.llm else []
        
    def execute(self, context) -> StageResult:
        if not self.llm:
            from plugins.local_llm import LocalLLMProvider
            self.llm = LocalLLMProvider()
            
        scene_manifest = None
        for node in context.execution_nodes:
            if isinstance(node.artifact, SceneManifest):
                scene_manifest = node.artifact
                break
                
        if not scene_manifest:
            raise ValueError("No SceneManifest found in context.")
            
        all_shots = []
        
        for scene in scene_manifest.scenes:
            # Generate deterministic shot_ids
            scene_hash = hashlib.sha256(scene.scene_id.encode('utf-8')).hexdigest()[:8]
            
            # Simulated Shot Expansion for Milestone 2 deterministic tests
            # (In Milestone 3, this calls the LLM with the prompt AST)
            shots = [
                Shot(shot_id=f"shot_{scene_hash}_01", duration=4.0),
                Shot(shot_id=f"shot_{scene_hash}_02", duration=3.5)
            ]
            all_shots.extend(shots)
            
        manifest = ShotManifest(
            shots=all_shots,
            generator="ShotPlannerStage",
            generator_version="0.1.0",
            schema_version="1.0"
        )
        
        node = ExecutionNode(artifact=manifest, stage_name="ShotPlannerStage")
        
        return StageResult(
            artifact=manifest,
            execution_node=node,
            metrics={"total_shots": len(all_shots)},
            metadata={}
        )
