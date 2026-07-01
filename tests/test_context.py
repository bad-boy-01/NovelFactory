import pytest
from core.pipeline.context import PipelineContext
from core.domain.project import ProjectManifest
from core.domain.bible import StoryBible
from core.domain.asset import Asset, AssetType
from core.domain.base import ProvenanceGraph
from pathlib import Path

def test_pipeline_context_creation():
    manifest = ProjectManifest(project_name="Test", dataset_id="Dataset")
    ctx = PipelineContext(project_manifest=manifest)
    
    assert ctx.project_manifest.project_name == "Test"
    assert ctx.story_bible is None
    assert ctx.current_chapter is None
    assert ctx.current_scene is None
    assert len(ctx.assets) == 0
    assert len(ctx.prompts) == 0
    assert len(ctx.state) == 0

def test_pipeline_context_mutation():
    manifest = ProjectManifest(project_name="Test", dataset_id="Dataset")
    ctx = PipelineContext(project_manifest=manifest)
    
    bible = StoryBible()
    ctx.story_bible = bible
    
    assert ctx.story_bible is not None
    assert ctx.story_bible.version == 1

def test_pipeline_context_assets():
    manifest = ProjectManifest(project_name="Test", dataset_id="Dataset")
    ctx = PipelineContext(project_manifest=manifest)
    
    asset = Asset(
        asset_type=AssetType.IMAGE,
        file_path=Path("/tmp/img.png"),
        hash_sha256="abc",
        provenance=ProvenanceGraph()
    )
    
    ctx.assets[asset.id] = asset
    assert len(ctx.assets) == 1
    assert ctx.assets[asset.id].hash_sha256 == "abc"
