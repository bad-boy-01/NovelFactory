from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.prompt.ast import PromptManifest
from core.domain.assets.registry import AssetRegistry, Asset
from core.rendering.render_queue import RenderQueue
import logging
import os

logger = logging.getLogger(__name__)

class DiffusionRendererStage(PipelineStage):
    def __init__(self, diffusion_provider=None):
        self.diffusion = diffusion_provider
        self.queue = RenderQueue()

    def get_providers(self) -> list:
        return [self.diffusion] if self.diffusion else []

    def execute(self, context) -> StageResult:
        if not self.diffusion:
            from plugins.local_diffusion import LocalDiffusionProvider
            self.diffusion = LocalDiffusionProvider()
            
        prompt_manifest = None
        for node in context.execution_nodes.values():
            if isinstance(node.artifact, PromptManifest):
                prompt_manifest = node.artifact
                break
                
        if not prompt_manifest:
            raise ValueError("DiffusionRenderer: Missing PromptManifest.")
            
        # 1. Populate the Render Queue with jobs
        for p in prompt_manifest.prompts:
            self.queue.add_job(
                job_id=p.prompt_id,
                scene_id=p.scene_id,
                shot_id=p.shot_id,
                prompt_hash=p.prompt_id, # Using ID as hash for simplicity
                seed=p.seed
            )
            
        registry = AssetRegistry(schema_version="1.0")
        
        # 2. Consume Render Queue
        pending_jobs = self.queue.get_pending_jobs(limit=100)
        
        os.makedirs("workspace", exist_ok=True)
        
        for job in pending_jobs:
            job_id = job["job_id"]
            self.queue.increment_attempts(job_id)
            
            # Find the corresponding AST
            target_prompt = next((p for p in prompt_manifest.prompts if p.prompt_id == job_id), None)
            if not target_prompt:
                self.queue.update_job_status(job_id, "FAILED")
                continue
                
            try:
                logger.info(f"Rendering job {job_id} (Seed {target_prompt.seed})")
                
                ast = target_prompt.ast
                prompt_str = f"{ast.subject}, {ast.environment}, {ast.camera.distance} {ast.camera.angle}, {ast.lighting}, {ast.composition}, {ast.style}"
                
                # Render using the provider
                image = self.diffusion.generate_image(
                    prompt=prompt_str,
                    negative_prompt=ast.negative,
                    num_inference_steps=target_prompt.steps,
                    guidance_scale=target_prompt.cfg,
                    seed=target_prompt.seed
                )
                
                output_path = f"workspace/{job_id}.png"
                image.save(output_path)
                
                self.queue.update_job_status(job_id, "COMPLETE", image_path=output_path)
                self.queue.log_event("DiffusionRenderer", f"Completed {job_id}")
                
                # Create Asset
                asset = Asset(
                    asset_id=f"asset_{target_prompt.shot_id}",
                    type="image",
                    checksum="dummy_checksum", # In real env, hash the file
                    prompt_hash=job_id,
                    seed=target_prompt.seed,
                    path=output_path
                )
                registry.assets[asset.asset_id] = asset
                
            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}", exc_info=True)
                self.queue.update_job_status(job_id, "FAILED")
                self.queue.log_event("DiffusionRenderer", f"Failed {job_id}", details=str(e))
                
        node = ExecutionNode(artifact=registry, stage_name="DiffusionRendererStage")
        
        return StageResult(
            artifact=registry,
            execution_node=node,
            metrics={"jobs_processed": len(pending_jobs)},
            metadata={}
        )
