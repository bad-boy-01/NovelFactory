import json
from pathlib import Path
from dataclasses import asdict
from core.pipeline.stage import StageResult
from core.domain.asset import ExecutionNode, FrameManifest, FrameManifestEntry
from plugins.interfaces import VideoRendererProvider

class RenderingStage:
    def __init__(self, renderer: VideoRendererProvider, output_file: str = "workspace/008_video.mp4"):
        self.renderer = renderer
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
    def get_providers(self) -> list:
        return [self.renderer]
        
    def execute(self, context) -> StageResult:
        frames = []
        frame_idx = 0
        
        for node in context.execution_nodes:
            from core.domain.asset import GeneratedImage
            if isinstance(node.artifact, GeneratedImage):
                entry = FrameManifestEntry(
                    frame_index=frame_idx,
                    beat_id=f"beat_{frame_idx}",
                    image_path=node.artifact.image_path,
                    prompt_hash=node.artifact.prompt_hash,
                    asset_id=f"asset_{frame_idx}"
                )
                frames.append(entry)
                frames.append(FrameManifestEntry(frame_index=frame_idx+1, beat_id=f"beat_{frame_idx+1}", image_path=node.artifact.image_path, prompt_hash=node.artifact.prompt_hash, asset_id=f"asset_{frame_idx}"))
                frames.append(FrameManifestEntry(frame_index=frame_idx+2, beat_id=f"beat_{frame_idx+2}", image_path=node.artifact.image_path, prompt_hash=node.artifact.prompt_hash, asset_id=f"asset_{frame_idx}"))
                frame_idx += 3
                
        manifest = FrameManifest(frames=frames)
        
        manifest_path = self.output_file.parent / "007_frame_manifest.json"
        with open(manifest_path, "w") as f:
            manifest_dict = [asdict(e) for e in manifest.frames]
            for d in manifest_dict:
                d["image_path"] = str(d["image_path"])
            json.dump(manifest_dict, f, indent=2)
            
        output_path = self.renderer.render_video(
            manifest=manifest, 
            audio_paths=[],
            output_path=self.output_file
        )
        
        node = ExecutionNode(artifact=output_path, stage_name="RenderingStage")
        
        return StageResult(
            artifact=output_path,
            execution_node=node,
            metrics={},
            metadata={"manifest_path": str(manifest_path)}
        )
