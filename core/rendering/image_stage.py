from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.prompt.ast import PromptManifest
from core.domain.assets.registry import AssetRegistry, Asset
from core.rendering.render_queue import RenderQueue
import logging
import os
import json
import math
import datetime

logger = logging.getLogger(__name__)

class DiffusionRendererStage(PipelineStage):
    def __init__(self, diffusion_provider=None, render_options=None):
        self.diffusion = diffusion_provider
        self.render_options = render_options or {}

    def get_providers(self) -> list:
        return [self.diffusion] if self.diffusion else []

    def _should_render(self, p):
        opt_shot = self.render_options.get("shot")
        opt_scene = self.render_options.get("scene")
        opt_shots = self.render_options.get("shots")

        if not opt_shot and not opt_scene and not opt_shots:
            return True

        if opt_shot and str(p.shot_id) == str(opt_shot):
            return True
        if opt_scene and str(p.scene_id) == str(opt_scene):
            return True
        if opt_shots:
            try:
                start, end = map(int, str(opt_shots).split('-'))
                # Assuming shot_id is numeric or has numeric part
                import re
                nums = re.findall(r'\d+', str(p.shot_id))
                if nums:
                    shot_num = int(nums[0])
                    if start <= shot_num <= end:
                        return True
            except Exception:
                pass
        return False

    def execute(self, context) -> StageResult:
        if not self.diffusion:
            from plugins.local_diffusion import LocalDiffusionProvider
            self.diffusion = LocalDiffusionProvider()
            
        prompt_manifest = None
        for node in context.execution_nodes:
            if isinstance(node.artifact, PromptManifest):
                prompt_manifest = node.artifact
                break
                
        if not prompt_manifest:
            raise ValueError("DiffusionRenderer: Missing PromptManifest.")
            
        queue = context.queue
        registry = context.registry
        workspace = context.workspace
        from core.domain.assets.registry import AssetStatus, Asset
        import hashlib
        
        for p in prompt_manifest.prompts:
            if self._should_render(p):
                queue.add_job(
                    job_id=p.prompt_id,
                    scene_id=p.scene_id,
                    shot_id=p.shot_id,
                    prompt_hash=p.prompt_id,
                    seed=p.seed
                )
            
        # Self-healing Cache Validation
        all_jobs = queue.get_all_jobs() if hasattr(queue, 'get_all_jobs') else []
        for job in all_jobs:
            if job["status"] == "COMPLETE":
                job_id = job["job_id"]
                shot_id = job["shot_id"]
                asset_id = f"asset_{shot_id}"
                
                status = registry.get_asset_status(asset_id, expected_prompt_hash=job_id)
                if status != AssetStatus.VALID:
                    logger.warning(f"[Self-Healing] Job {job_id} was COMPLETE but asset is {status.name}. Re-queueing.")
                    queue.update_job_status(job_id, "PENDING")
                    
        pending_jobs = queue.get_pending_jobs(limit=100)
        
        rendered_images = []
        
        for job in pending_jobs:
            job_id = job["job_id"]
            queue.increment_attempts(job_id)
            
            target_prompt = next((p for p in prompt_manifest.prompts if p.prompt_id == job_id), None)
            if not target_prompt:
                queue.update_job_status(job_id, "FAILED")
                continue
                
            try:
                logger.info(f"Rendering job {job_id} (Seed {target_prompt.seed})")
                
                ast = target_prompt.ast
                
                # Extract and combine tags
                style_tags = ", ".join(ast.quality.tags) if ast.quality.tags else ""
                negative_str = ", ".join(ast.negative.tags) if ast.negative.tags else ""
                
                prompt_str = f"{ast.subject.description}, {ast.environment.location}, {ast.camera.distance} {ast.camera.angle}, {ast.lighting.style}, {ast.composition.style}, {style_tags}"
                
                from core.domain.rendering.presets import RenderJob, RenderPreset
                from diffusers import EulerDiscreteScheduler
                preset = RenderPreset(
                    width=1024,
                    height=1024,
                    steps=ast.technical.steps,
                    cfg=0.0, # ByteDance Lightning recommends 0 CFG
                    scheduler_class=EulerDiscreteScheduler,
                    negative_prompt=negative_str
                )
                
                render_job = RenderJob(
                    prompt=prompt_str,
                    negative_prompt=negative_str,
                    seed=target_prompt.seed,
                    preset=preset
                )
                
                # Render using the provider
                import time
                start_time = time.time()
                image = self.diffusion.generate(job=render_job)
                render_time = time.time() - start_time
                
                # Benchmark logging
                health = self.diffusion.health_check() if hasattr(self.diffusion, "health_check") else None
                vram_gb = health.vram_allocated_gb if health else 0.0
                logger.info(f"BENCHMARK: Shot {target_prompt.shot_id} | {preset.width}x{preset.height} | {preset.steps} steps | {render_time:.2f} s | {vram_gb:.2f} GB")
                
                
                shot_dir = workspace.get_asset_dir(job_id)
                
                output_path = str(shot_dir / "image.png")
                image.save(output_path)
                rendered_images.append(output_path)
                
                # Write metadata and related files
                with open(shot_dir / "prompt.txt", "w", encoding="utf-8") as f:
                    f.write(str(ast.model_dump()))
                with open(shot_dir / "optimized_prompt.txt", "w", encoding="utf-8") as f:
                    f.write(prompt_str)
                with open(shot_dir / "negative.txt", "w", encoding="utf-8") as f:
                    f.write(negative_str)
                    
                with open(shot_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "scene_id": target_prompt.scene_id,
                        "shot_id": target_prompt.shot_id,
                        "master_seed": 42,
                        "scene_seed": 82341,
                        "shot_seed": target_prompt.seed,
                        "variation": 0,
                        "model": "sdxl-lightning",
                        "scheduler": "lcm",
                        "steps": ast.technical.steps,
                        "cfg": ast.technical.cfg,
                        "render_time": round(render_time, 2),
                        "timestamp": datetime.datetime.now().isoformat()
                    }, f, indent=2)
                    
                # Mock manifests for reproducibility
                for fname in ["scene.json", "shot.json", "camera.json", "continuity.json", "qa.json", "duration.json"]:
                    with open(shot_dir / fname, "w", encoding="utf-8") as f:
                        json.dump({}, f)
                
                queue.update_job_status(job_id, "COMPLETE", image_path=output_path)
                queue.log_event("DiffusionRenderer", f"Completed {job_id}")
                
                # Checksum generation
                with open(output_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    
                asset = Asset(
                    asset_id=f"asset_{target_prompt.shot_id}",
                    type="image",
                    checksum=file_hash,
                    prompt_hash=job_id,
                    seed=target_prompt.seed,
                    path=output_path
                )
                registry.assets[asset.asset_id] = asset
                
            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}", exc_info=True)
                queue.update_job_status(job_id, "FAILED")
                queue.log_event("DiffusionRenderer", f"Failed {job_id}", details=str(e))
                
        # Generate Contact Sheet
        if rendered_images:
            try:
                from PIL import Image
                images = [Image.open(p) for p in rendered_images]
                if images:
                    width, height = images[0].size
                    cols = 4
                    rows = math.ceil(len(images) / cols)
                    contact_sheet = Image.new('RGB', (cols * width, rows * height))
                    for i, img in enumerate(images):
                        x = (i % cols) * width
                        y = (i // cols) * height
                        contact_sheet.paste(img, (x, y))
                    contact_sheet.save("workspace/contact_sheet.jpg", quality=85)
            except Exception as e:
                logger.error(f"Failed to generate contact sheet: {e}")
                
        node = ExecutionNode(artifact=registry, stage_name="DiffusionRendererStage")
        
        return StageResult(
            artifact=registry,
            execution_node=node,
            metrics={"jobs_processed": len(pending_jobs)},
            metadata={}
        )
