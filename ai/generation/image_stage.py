from pathlib import Path
from core.pipeline.stage import StageResult
from core.domain.asset import ExecutionNode, GenerationRequest
from plugins.interfaces import ImageGeneratorProvider
from core.pipeline.cache import CacheProvider
import hashlib

class ImageGenerationStage:
    def __init__(self, provider: ImageGeneratorProvider, cache: CacheProvider, output_dir: str = ".output/frames"):
        self.provider = provider
        self.cache = cache
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_providers(self) -> list:
        return [self.provider]

    def execute(self, context) -> StageResult:
        ast = context.state.get("PromptAST")
        if not ast:
            raise ValueError("No PromptAST found in context state.")
            
        compiled_prompt = f"{ast.character}, {ast.outfit}, {ast.scene}, {ast.camera}, {ast.lighting}, {ast.style}"
        prompt_hash = hashlib.sha256(compiled_prompt.encode()).hexdigest()[:16]
        seed = 42
        
        output_filename = self.output_dir / f"frame_{prompt_hash[:8]}.png"
        
        request = GenerationRequest(
            compiled_prompt=compiled_prompt,
            negative_prompt=ast.negative or "",
            seed=seed,
            prompt_hash=prompt_hash,
            model_id=self.provider.get_model_name(),
            output_path=output_filename,
            width=self.provider.config.width,
            height=self.provider.config.height,
            steps=self.provider.config.steps,
            guidance_scale=self.provider.config.guidance_scale
        )
        
        cache_key = f"{prompt_hash}_{seed}_{request.model_id}"
        cached = self.cache.check_cache(cache_key) if hasattr(self.cache, 'check_cache') else None
        
        if cached:
            generated_image = cached
            cache_hit = True
        else:
            generated_image = self.provider.generate_image(request)
            cache_hit = False
            
        node = ExecutionNode(
            artifact=generated_image,
            request=request,
            stage_name="ImageGenerationStage",
            cache_key=cache_key
        )
        
        return StageResult(
            artifact=generated_image,
            execution_node=node,
            metrics={"cache_hit": cache_hit},
            metadata={}
        )
