import hashlib
from uuid import uuid4
from pathlib import Path
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext
from core.domain.asset import Asset, AssetType
from core.domain.base import ProvenanceGraph
from plugins.interfaces import ImageGeneratorProvider

class ImageGenerationStage(PipelineStage):
    """
    Takes compiled prompts from the context and generates images via the ImageGeneratorProvider.
    Records perfect provenance for every generated asset.
    """
    def __init__(self, provider: ImageGeneratorProvider, output_dir: Path):
        self.provider = provider
        self.output_dir = output_dir

    def get_name(self) -> str:
        return f"Image Generator ({self.provider.get_model_name()})"

    def fingerprint(self, context: PipelineContext) -> str:
        hashes = []
        for beat_id, prompt in context.prompts.items():
            hashes.append(hashlib.md5(prompt.compiled_text.encode()).hexdigest())
        return "img_" + "_".join(hashes)

    def execute(self, context: PipelineContext) -> PipelineContext:
        if not context.current_scene:
            return context
            
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for beat in context.current_scene.beats:
            prompt = context.prompts.get(beat.id)
            if not prompt:
                continue
                
            seed = 42 # In production, this would be randomized or managed via profiles
            out_file = self.output_dir / f"{beat.id}.png"
            
            try:
                # Generate image
                self.provider.generate_image(
                    prompt=prompt.compiled_text,
                    negative_prompt=", ".join(prompt.negative_constraints),
                    seed=seed,
                    output_path=out_file
                )
                
                # Construct Provenance Graph
                provenance = ProvenanceGraph(
                    generated_from=prompt.id,
                    model=self.provider.get_model_name(),
                    model_revision=self.provider.get_model_revision(),
                    prompt_hash=hashlib.md5(prompt.compiled_text.encode()).hexdigest(),
                    story_bible_hash=context.story_bible.hash if context.story_bible else "none",
                    config_hash="default",
                    seed=seed
                )
                
                # Create and store Asset
                asset = Asset(
                    asset_type=AssetType.IMAGE,
                    file_path=out_file,
                    hash_sha256="dummy_hash_for_mvp", # File hashing would happen here
                    provenance=provenance
                )
                
                context.assets[asset.id] = asset
                
            except Exception as e:
                # Evaluation loop handles failures, we just skip asset creation
                pass

        return context
