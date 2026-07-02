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

class SDXLLightningProvider(ImageGenerationProvider):
    def __init__(self, config: DiffusionConfig = None):
        self.config = config or DiffusionConfig()
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load(self) -> None:
        if self.pipeline is not None:
            logger.info("[Resource] SDXL Lightning already resident in VRAM.")
            return
            
        logger.info("[Resource] Loading SDXL Lightning into VRAM...")
        
        # We use a 4-step UNet from ByteDance for SDXL Lightning
        unet = UNet2DConditionModel.from_pretrained(
            "ByteDance/SDXL-Lightning",
            subfolder="unet",
            torch_dtype=self.config.dtype,
            cache_dir=self.config.cache_dir
        )
        
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.config.model_id,
            unet=unet,
            torch_dtype=self.config.dtype,
            variant="fp16",
            use_safetensors=True,
            cache_dir=self.config.cache_dir
        )
        
        if self.device == "cuda":
            if self.config.cpu_offload:
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline.to("cuda")
                
        # Warmup
        logger.info("[Resource] Warming up SDXL Lightning...")
        preset = self._get_warmup_preset()
        self._generate_internal("warmup", "", 42, preset)
        
    def _get_warmup_preset(self):
        from core.domain.rendering.presets import RenderPreset
        return RenderPreset(width=256, height=256, steps=1, cfg=0.0)
        
    def _generate_internal(self, prompt, negative, seed, preset, callback=None):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Instantiate scheduler from class
        self.pipeline.scheduler = preset.scheduler_class.from_config(
            self.pipeline.scheduler.config, 
            timestep_spacing="trailing"
        )
        
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
        logger.info(f"SDXL Lightning Generating: {job.prompt[:40]}...")
        return self._generate_internal(job.prompt, job.negative_prompt, job.seed, job.preset, callback)
        
    def health_check(self) -> ProviderHealth:
        loaded = self.pipeline is not None
        vram = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        return ProviderHealth(
            loaded=loaded, 
            device=self.device, 
            model="SDXL-Lightning", 
            dtype=str(self.config.dtype), 
            vram_allocated_gb=vram
        )
        
    def unload(self) -> None:
        logger.info("[Resource] Unloading SDXL Lightning...")
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        from core.utils.vram import flush_vram
        flush_vram("Diffusion unloaded")

LocalDiffusionProvider = SDXLLightningProvider
