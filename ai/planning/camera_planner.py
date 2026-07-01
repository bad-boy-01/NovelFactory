from core.pipeline.stage import PipelineStage, StageResult
from core.domain.asset import ExecutionNode
from core.domain.scene import ShotManifest
import copy

class CameraPlannerStage(PipelineStage):
    def get_providers(self) -> list:
        return []
        
    def execute(self, context) -> StageResult:
        shot_manifest = None
        for node in context.execution_nodes.values():
            if isinstance(node.artifact, ShotManifest):
                shot_manifest = node.artifact
                break
                
        if not shot_manifest:
            raise ValueError("No ShotManifest found in context.")
            
        # Deterministically apply camera logic without LLM (Fast, cheap, stable)
        enriched_manifest = copy.deepcopy(shot_manifest)
        
        for i, shot in enumerate(enriched_manifest.shots):
            shot.camera_type = "cinematic"
            shot.lens = "50mm" if i % 2 == 0 else "85mm"
            shot.angle = "eye-level"
            shot.distance = "medium" if i % 2 == 0 else "close-up"
            shot.movement = "slow pan right" if i % 2 == 0 else "static"
            
        node = ExecutionNode(artifact=enriched_manifest, stage_name="CameraPlannerStage")
        
        return StageResult(
            artifact=enriched_manifest,
            execution_node=node,
            metrics={"shots_planned": len(enriched_manifest.shots)},
            metadata={}
        )
