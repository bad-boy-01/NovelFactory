import hashlib
from pathlib import Path
from typing import List
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext
from core.domain.asset import Asset, AssetType
from core.domain.base import ProvenanceGraph
from plugins.interfaces import VideoRendererProvider

class RenderingStage(PipelineStage):
    """
    Final stage: assembles generated images into a sequence and renders an MP4.
    """
    def __init__(self, renderer: VideoRendererProvider, output_dir: Path):
        self.renderer = renderer
        self.output_dir = output_dir

    def get_name(self) -> str:
        return "Video Rendering"

    def fingerprint(self, context: PipelineContext) -> str:
        hashes = [asset.hash_sha256 for asset in context.assets.values() if asset.asset_type == AssetType.IMAGE]
        return "vid_" + hashlib.md5("".join(hashes).encode()).hexdigest()

    def execute(self, context: PipelineContext) -> PipelineContext:
        if not context.current_chapter:
            return context
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect ordered images based on chapter -> scene -> beat -> prompt -> image asset
        ordered_images = []
        for scene in context.current_chapter.scenes:
            for beat in scene.beats:
                prompt = context.prompts.get(beat.id)
                if prompt:
                    for asset in context.assets.values():
                        if asset.asset_type == AssetType.IMAGE and asset.provenance.generated_from == prompt.id:
                            # Verify it passed evaluation
                            if not asset.evaluation or not asset.evaluation.retry_needed:
                                ordered_images.append(asset.file_path)
                            break
        
        if not ordered_images:
            return context
            
        out_file = self.output_dir / "chapter_output.mp4"
        
        try:
            self.renderer.render_video(
                image_paths=ordered_images,
                audio_paths=[], 
                output_path=out_file
            )
            
            provenance = ProvenanceGraph(
                model="VideoRenderer",
                config_hash="default"
            )
            
            video_asset = Asset(
                asset_type=AssetType.VIDEO,
                file_path=out_file,
                hash_sha256="dummy_hash_for_mvp",
                provenance=provenance
            )
            
            context.assets[video_asset.id] = video_asset
        except Exception as e:
            pass
            
        return context
