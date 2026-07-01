import pytest
from pathlib import Path
from typing import List
from core.pipeline.context import PipelineContext
from core.domain.project import ProjectManifest
from core.domain.story import Chapter, Scene, Beat
from core.domain.prompt import DeclarativePrompt
from core.domain.asset import Asset, AssetType
from core.domain.base import ProvenanceGraph
from ai.generation.rendering import RenderingStage
from plugins.interfaces import VideoRendererProvider

class MockVideoRenderer(VideoRendererProvider):
    def render_video(self, image_paths: List[Path], audio_paths: List[Path], output_path: Path) -> Path:
        output_path.write_text("fake video data")
        return output_path

def test_rendering_stage(tmp_path):
    manifest = ProjectManifest(project_name="Test", dataset_id="Dataset")
    ctx = PipelineContext(project_manifest=manifest)
    
    beat1 = Beat(text="Beat 1")
    beat2 = Beat(text="Beat 2")
    scene = Scene(beats=[beat1, beat2])
    ctx.current_chapter = Chapter(title="Chap 1", scenes=[scene])
    
    dp1 = DeclarativePrompt()
    dp2 = DeclarativePrompt()
    ctx.prompts[beat1.id] = dp1
    ctx.prompts[beat2.id] = dp2
    
    asset1 = Asset(
        asset_type=AssetType.IMAGE,
        file_path=Path("/tmp/img1.png"),
        hash_sha256="h1",
        provenance=ProvenanceGraph(generated_from=dp1.id)
    )
    asset2 = Asset(
        asset_type=AssetType.IMAGE,
        file_path=Path("/tmp/img2.png"),
        hash_sha256="h2",
        provenance=ProvenanceGraph(generated_from=dp2.id)
    )
    ctx.assets[asset1.id] = asset1
    ctx.assets[asset2.id] = asset2
    
    renderer = MockVideoRenderer()
    out_dir = tmp_path / "exports"
    stage = RenderingStage(renderer=renderer, output_dir=out_dir)
    
    assert stage.get_name() == "Video Rendering"
    
    ctx = stage.execute(ctx)
    
    # We started with 2 images, now we should have 3 assets (2 images + 1 video)
    assert len(ctx.assets) == 3
    
    video_assets = [a for a in ctx.assets.values() if a.asset_type == AssetType.VIDEO]
    assert len(video_assets) == 1
    
    video_asset = video_assets[0]
    assert video_asset.file_path.exists()
    assert video_asset.file_path.read_text() == "fake video data"
    assert video_asset.provenance.model == "VideoRenderer"
