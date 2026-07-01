import json
from pathlib import Path
from dataclasses import asdict
from core.pipeline.context import PipelineContext
from plugins.interfaces import VideoRendererProvider
from core.domain.asset import FrameManifest, FrameManifestEntry

class RenderingStage:
    """
    Constructs the sequenced FrameManifest and delegates to the Video Renderer.
    """
    def __init__(self, renderer: VideoRendererProvider, output_file: str = ".output/final.mp4"):
        self.renderer = renderer
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
    def execute(self, context: PipelineContext):
        # Build the ordered manifest from the context assets
        frames = []
        assets = getattr(context, 'artifacts', []) # SequentialExecutor commits to context.artifacts
        
        frame_idx = 0
        for asset in assets:
            if hasattr(asset, 'generated_image') and asset.generated_image:
                entry = FrameManifestEntry(
                    frame_index=frame_idx,
                    beat_id=f"beat_{frame_idx}",
                    image_path=asset.generated_image.image_path,
                    prompt_hash=asset.generated_image.prompt_hash,
                    asset_id=asset.asset_id
                )
                frames.append(entry)
                frame_idx += 1
                
        manifest = FrameManifest(frames=frames)
        
        # Save manifest for observability
        manifest_path = self.output_file.parent / "frames_manifest.json"
        with open(manifest_path, "w") as f:
            manifest_dict = [asdict(e) for e in manifest.frames]
            # Convert paths to strings for JSON
            for d in manifest_dict:
                d["image_path"] = str(d["image_path"])
            json.dump(manifest_dict, f, indent=2)
        
        # Render
        output_path = self.renderer.render_video(
            manifest=manifest, 
            audio_paths=[],
            output_path=self.output_file
        )
        
        return {
            "video_path": output_path,
            "manifest_path": manifest_path
        }
