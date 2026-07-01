import random
import time
import uuid
import logging
from typing import Dict, Any, List
from pathlib import Path

from plugins.interfaces import LLMProvider, ImageGeneratorProvider, EvaluatorPlugin, VideoRendererProvider
from core.pipeline.context import PipelineContext
from core.domain.asset import Asset, EvaluationResult

logger = logging.getLogger(__name__)

class MockLLMProvider(LLMProvider):
    def generate_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        logger.info({"provider": "MockLLMProvider", "event": "generate_json", "prompt_len": len(prompt)})
        
        # Introduce "trace realism" - occasionally drop color_palette
        drop_color = random.random() < 0.2
        
        chars = [
            {
                "name": "Alice",
                "visual_dna": "young woman, blonde hair, blue eyes",
                "outfit": "school uniform with a red tie",
                "color_palette": "blue, red, white" if not drop_color else ""
            },
            {
                "name": "Bob",
                "visual_dna": "tall man, dark hair, glasses",
                "outfit": "casual suit",
                "color_palette": "black, grey"
            }
        ]
        
        # Occasionally miss a secondary character
        if random.random() < 0.1:
            logger.warning({"provider": "MockLLMProvider", "event": "hallucination", "detail": "Dropped character Bob"})
            chars = [chars[0]]
            
        return {"characters": chars}

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        return "Mock response text."


class MockImageGenerator(ImageGeneratorProvider):
    def get_model_name(self) -> str:
        return "mock-diffusion-v1"

    def get_model_revision(self) -> str:
        return "1.0.0-mock"

    def generate_image(self, prompt: str, negative_prompt: str, seed: int, output_path: Path) -> Path:
        logger.info({"provider": "MockImageGenerator", "event": "generate_image", "seed": seed, "output": str(output_path)})
        time.sleep(0.1) # Simulate some work
        
        # Simulate occasional corruption
        if random.random() < 0.05:
            logger.error({"provider": "MockImageGenerator", "event": "corruption", "detail": "Simulated image generation failure"})
            raise RuntimeError("CUDA out of memory (simulated)")

        # Create a dummy image file with varying size
        size_bytes = random.randint(50000, 150000)
        with open(output_path, 'wb') as f:
            f.write(os.urandom(100) if random.random() < 0.01 else b'0' * size_bytes) # Occasional invalid small file
            
        return output_path


class MockEvaluator(EvaluatorPlugin):
    def __init__(self, name: str = "MockConsistencyEvaluator"):
        self.name = name

    def get_name(self) -> str:
        return self.name

    def evaluate(self, asset: Asset, context: PipelineContext) -> EvaluationResult:
        logger.info({"provider": "MockEvaluator", "event": "evaluate", "asset": str(asset.file_path)})
        
        # Check for simulated corruption
        if asset.file_path.exists() and asset.file_path.stat().st_size < 1000:
            return EvaluationResult(score=0.1, reason="Image file is corrupted or too small", retry_needed=True)

        # Vary score to simulate borderline cases
        score = random.uniform(0.65, 1.0)
        
        retry = False
        reason = "Pass"
        if score < 0.8:
            retry = True
            reason = "Identity drift detected (simulated)"
            logger.warning({"provider": "MockEvaluator", "event": "failure", "score": score, "reason": reason})

        return EvaluationResult(score=score, reason=reason, retry_needed=retry)


class MockVideoRenderer(VideoRendererProvider):
    def render_video(self, image_paths: List[Path], audio_paths: List[Path], output_path: Path) -> Path:
        logger.info({"provider": "MockVideoRenderer", "event": "render_video", "frames": len(image_paths)})
        time.sleep(0.2) # Simulate encoding delay
        
        if random.random() < 0.02:
            raise RuntimeError("FFmpeg encoding failed (simulated)")

        with open(output_path, 'wb') as f:
            f.write(b'fake_video_data')
            
        return output_path

import os
