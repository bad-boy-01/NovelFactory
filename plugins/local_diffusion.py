import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from plugins.interfaces import DiffusionConfig
import logging

logger = logging.getLogger(__name__)

class ImageGenerationProvider:
    def generate_image(self, prompt: str, negative_prompt: str, num_inference_steps: int, guidance_scale: float, seed: int) -> Image.Image:
        raise NotImplementedError

class SD15Provider(ImageGenerationProvider):
    def __init__(self, config: DiffusionConfig = None):
        self.config = config or DiffusionConfig()
        
    def generate_image(self, prompt: str, negative_prompt: str, num_inference_steps: int, guidance_scale: float, seed: int) -> Image.Image:
        logger.info(f"SD15 Generating: {prompt[:30]}...")
        img = Image.new('RGB', (512, 512), color='blue')
        return img

class SDXLProvider(ImageGenerationProvider):
    def __init__(self, config: DiffusionConfig = None):
        self.config = config or DiffusionConfig()
        
    def generate_image(self, prompt: str, negative_prompt: str, num_inference_steps: int, guidance_scale: float, seed: int) -> Image.Image:
        logger.info(f"SDXL Generating (High Fidelity): {prompt[:30]}...")
        img = Image.new('RGB', (1024, 1024), color='green')
        return img

LocalDiffusionProvider = SD15Provider
