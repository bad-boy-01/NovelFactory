import torch
import importlib.metadata
import datetime
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline
        print(f"[DIFFUSION] Initializing pipeline config for {self.config.model_id}...")

    def load(self):
        print(f"[DIFFUSION] Loading {self.config.model_id} into VRAM...")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            torch_dtype=self.config.dtype
        )
        if self.config.cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to("cuda")

    def unload(self):
        from core.utils.vram import flush_vram
        print("[DIFFUSION] Unloading model...")
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        flush_vram("Diffusion unloaded")

    def shutdown(self):
        print("[DIFFUSION] Shutting down...")

    def generate_image(self, request: 'GenerationRequest') -> GeneratedImage:
        if not self.pipeline:
            self.load()

        # Build generator
        generator = torch.Generator(device="cpu").manual_seed(request.seed)
        
        # Execute Generation
        output = self.pipeline(
            prompt=request.compiled_prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            generator=generator
        )
        image = output.images[0]
        
        # Persist physical file
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(request.output_path)
        
        # Construct provenance metadata
        diff_ver = importlib.metadata.version('diffusers') if importlib.util.find_spec('diffusers') else "unknown"
        torch_ver = importlib.metadata.version('torch') if importlib.util.find_spec('torch') else "unknown"
        scheduler_name = self.pipeline.scheduler.__class__.__name__

        provenance = ProvenanceRecord(
            model_id=request.model_id,
            revision=self.config.revision,
            prompt_hash=request.prompt_hash,
            seed=request.seed,
            scheduler=scheduler_name,
            guidance_scale=request.guidance_scale,
            inference_steps=request.steps,
            diffusers_version=diff_ver,
            torch_version=torch_ver
        )
        
        # Ensure VRAM is cleared if using sequential models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return GeneratedImage(
            image_path=request.output_path,
            width=request.width,
            height=request.height,
            seed=request.seed,
            prompt_hash=request.prompt_hash,
            model_id=request.model_id,
            cache_hit=False,
            provenance=provenance
        )
