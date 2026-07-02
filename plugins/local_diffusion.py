import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from PIL import Image
from plugins.interfaces import DiffusionConfig, ImageGenerationProvider, ProviderHealth
from core.domain.rendering.presets import RenderJob
import logging
import gc

logger = logging.getLogger(__name__)

class MockProvider(ImageGenerationProvider):
    def __init__(self, config: DiffusionConfig = None):
        self.config = config or DiffusionConfig()
        
    def load(self) -> None:
        pass
        
    def generate(self, job: RenderJob, callback=None) -> Image.Image:
        logger.info(f"Mock Generating: {job.prompt[:30]}...")
        if callback:
            for i in range(job.preset.steps):
                callback(i, job.preset.steps)
        return Image.new('RGB', (job.preset.width, job.preset.height), color='green')
        
    def health_check(self) -> ProviderHealth:
        return ProviderHealth(loaded=True, device="cpu", model="mock", dtype="none", vram_allocated_gb=0.0)
        
    def unload(self) -> None:
        pass

class DiffusersProvider(ImageGenerationProvider):
    def __init__(self, config: DiffusionConfig = None):
        # We assume config is actually a ModelConfig in the new architecture
        self.config = config
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def capabilities(self):
        from plugins.interfaces import ProviderCapability
        return ProviderCapability(image=True, lora=bool(self.config.adapter))
        
    def load(self) -> None:
        if self.pipeline is not None:
            logger.info(f"[Resource] Model {self.config.model_id} already resident in VRAM.")
            return
            
        logger.info(f"[Resource] Loading {self.config.model_id} into VRAM...")
        
        unet = None
        if self.config.adapter and "SDXL-Lightning" in self.config.adapter:
            # Handle specific adapter logic
            unet = UNet2DConditionModel.from_pretrained(
                self.config.adapter,
                subfolder="unet",
                torch_dtype=getattr(torch, self.config.dtype, torch.float16),
                cache_dir=self.config.cache_dir
            )
        
        # Load the base model
        if unet:
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.config.model_id,
                unet=unet,
                torch_dtype=getattr(torch, self.config.dtype, torch.float16),
                variant="fp16",
                use_safetensors=True,
                cache_dir=self.config.cache_dir
            )
        else:
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=getattr(torch, self.config.dtype, torch.float16),
                variant="fp16",
                use_safetensors=True,
                cache_dir=self.config.cache_dir
            )
            
        if self.device == "cuda":
            if self.config.cpu_offload:
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline.to("cuda")
                
        self.warmup()
        
    def warmup(self) -> None:
        logger.info("[Resource] Warming up model...")
        from core.domain.rendering.presets import RenderPreset
        preset = RenderPreset(width=256, height=256, steps=1, cfg=0.0)
        self._generate_internal("warmup", "", 42, preset)
        
    def _generate_internal(self, prompt, negative, seed, preset, callback=None):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Instantiate scheduler from string
        if preset.sampler == "euler":
            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.pipeline.scheduler.config, 
                timestep_spacing="trailing"
            )
        # We can add more samplers later
        
        def cb_wrapper(step, timestep, latents):
            if callback:
                callback(step, preset.steps)
                
        image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=preset.steps,
            guidance_scale=preset.cfg,
            width=preset.width,
            height=preset.height,
            generator=generator,
            callback=cb_wrapper if callback else None,
            callback_steps=1
        ).images[0]
        
        return image
        
    def generate(self, job: RenderJob, callback=None) -> Image.Image:
        logger.info(f"Generating ({self.config.model_id}): {job.prompt[:40]}...")
        return self._generate_internal(job.prompt, job.negative_prompt, job.seed, job.preset, callback)
        
    def health_check(self) -> ProviderHealth:
        loaded = self.pipeline is not None
        vram = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        return ProviderHealth(
            loaded=loaded, 
            device=self.device, 
            model=self.config.model_id, 
            dtype=self.config.dtype, 
            vram_allocated_gb=vram
        )
        
    def unload(self) -> None:
        logger.info(f"[Resource] Unloading {self.config.model_id}...")
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        from core.utils.vram import flush_vram
        flush_vram("Diffusion unloaded")

LocalDiffusionProvider = DiffusersProvider
