from pathlib import Path
from core.pipeline.context import PipelineContext
from plugins.interfaces import ImageGeneratorProvider
from core.pipeline.cache import CacheProvider
from core.domain.asset import Asset


class ImageGenerationStage:
    """
    Consumes compiled AST prompts and orchestrates execution via the Cache
    or the underlying ImageGeneratorProvider. Does not know about SD directly.
    """
    def __init__(self, provider: ImageGeneratorProvider, cache: CacheProvider, output_dir: str = ".output/frames"):
        self.provider = provider
        self.cache = cache
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, context: PipelineContext):
        ast = getattr(context, 'current_ast', None)
        compiled_prompt = getattr(context, 'current_prompt', "")
        
        if not ast:
            raise ValueError("No PromptAST found in context.")
            
        prompt_hash = ast.to_canonical_hash()
        seed = getattr(context, 'current_seed', 42)
        model_id = self.provider.get_model_name()
        
        # 1. Lookup Cache
        cache_key = self.cache.generate_key(prompt_hash, seed, model_id) if hasattr(self.cache, 'generate_key') else f"{prompt_hash}_{seed}"
        cached_image = self.cache.check_cache(cache_key)
        
        if cached_image:
            generated_image = cached_image
        else:
            # 2. Cache Miss -> Execute Generation
            output_filename = self.output_dir / f"frame_{prompt_hash[:8]}.png"
            negative = ast.negative if ast.negative else ""
            
            from core.domain.asset import GenerationRequest
            request = GenerationRequest(
                compiled_prompt=compiled_prompt,
                negative_prompt=negative,
                seed=seed,
                prompt_hash=prompt_hash,
                model_id=self.provider.get_model_name(),
                output_path=output_filename,
                width=self.provider.config.width,
                height=self.provider.config.height,
                steps=self.provider.config.steps,
                guidance_scale=self.provider.config.guidance_scale
            )
            
            generated_image = self.provider.generate_image(request)
            
            # Note: Hard Contracts actually run in the Executor layer *after* this stage returns.
            # But wait, the generated image needs to be vetted before it's saved to cache/asset.
            # We return it, Executor runs Contracts, if it passes, Executor commits it.
            # To ensure invalid images don't enter cache, we should ONLY cache it after contracts.
            # But the executor only knows about PipelineContext right now. 
            # We will attach the unverified image to context as pending.
            
        # 3. Output to context
        # We simulate appending it as a potential asset
        asset = Asset(asset_id=f"asset_{prompt_hash[:8]}")
        asset.generated_image = generated_image
        
        # A full system would have a transactional commit hook, but for now we set it as current output
        # so the ContractEngine can validate it.
        context.pending_asset = asset
        
        return {
            "identity_blob": getattr(context, "story_bible", {}),
            "generated_image": generated_image
        }
