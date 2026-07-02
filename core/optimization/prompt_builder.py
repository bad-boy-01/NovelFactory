from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.prompt.ast import (
    PromptManifest, PromptManifestEntry, PromptAST, CameraAST,
    SubjectAST, EnvironmentAST, LightingAST, CompositionAST, QualityAST, NegativeAST
)
from core.domain.scene.manifest import ShotManifest
from core.domain.story.bible import StoryBible
import hashlib

class PromptBuilderStage(PipelineStage):
    def get_providers(self) -> list:
        return []

    def execute(self, context) -> StageResult:
        shot_manifest = None
        bible = None
        
        for node in context.execution_nodes:
            if isinstance(node.artifact, ShotManifest):
                shot_manifest = node.artifact
            elif isinstance(node.artifact, StoryBible):
                bible = node.artifact
                
        if not shot_manifest:
            raise ValueError("No ShotManifest found in context.")
            
        prompts = []
        for shot in shot_manifest.shots:
            # Deterministic seed based on hash
            seed_hash = hashlib.md5(shot.shot_id.encode('utf-8')).hexdigest()
            seed = int(seed_hash, 16) % (2**32 - 1)
            
            ast = PromptAST(
                subject=SubjectAST(description="Korean Manhwa scene"),
                characters=[],
                environment=EnvironmentAST(location="Determined by scene"),
                camera=CameraAST(
                    type=shot.camera_type,
                    lens=shot.lens,
                    angle=shot.angle,
                    distance=shot.distance,
                    movement=shot.movement
                ),
                lighting=LightingAST(style="soft cinematic lighting"),
                composition=CompositionAST(style="rule of thirds"),
                quality=QualityAST(tags=["anime cinematic", "high detail"]),
                negative=NegativeAST(tags=["low quality", "blurry", "distorted"])
            )
            
            entry = PromptManifestEntry(
                prompt_id=f"prompt_{seed_hash[:8]}",
                scene_id="unknown", # normally we'd pass scene_id down via Shot object
                shot_id=shot.shot_id,
                ast=ast,
                seed=seed
            )
            prompts.append(entry)
            
        manifest = PromptManifest(
            prompts=prompts,
            generator="PromptBuilderStage",
            generator_version="0.2.0"
        )
        
        node = ExecutionNode(artifact=manifest, stage_name="PromptBuilderStage")
        
        return StageResult(
            artifact=manifest,
            execution_node=node,
            metrics={"prompts_generated": len(prompts)},
            metadata={}
        )
