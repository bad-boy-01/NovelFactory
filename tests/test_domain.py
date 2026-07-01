import pytest
from pathlib import Path
from core.domain import ProvenanceGraph, Asset, AssetType, Beat, Scene, ProjectManifest, StoryBible, CharacterReference
from core.domain.story import VisualScreenplay, SceneDirectives
from core.domain.prompt import DeclarativePrompt

def test_provenance_graph():
    pg = ProvenanceGraph(model="flux", model_revision="v1")
    assert pg.model == "flux"
    assert pg.model_revision == "v1"
    assert pg.seed is None

def test_asset_creation():
    pg = ProvenanceGraph()
    asset = Asset(
        asset_type=AssetType.IMAGE,
        file_path=Path("/tmp/img.png"),
        hash_sha256="abc",
        provenance=pg
    )
    assert asset.asset_type == AssetType.IMAGE
    assert asset.id is not None
    assert asset.created_at is not None

def test_story_hierarchy():
    vs = VisualScreenplay(mood="lonely", camera="close-up")
    beat = Beat(text="He walked away.", visual_screenplay=vs)
    directives = SceneDirectives(keep_same_outfit=True)
    scene = Scene(beats=[beat], time_of_day="night", directives=directives)
    
    assert scene.time_of_day == "night"
    assert len(scene.beats) == 1
    assert scene.beats[0].visual_screenplay.camera == "close-up"
    assert scene.directives.keep_same_outfit is True

def test_declarative_prompt():
    dp = DeclarativePrompt(
        characters={"Alice": "red dress"},
        camera="wide shot",
        style="anime"
    )
    assert dp.camera == "wide shot"
    assert "Alice" in dp.characters

def test_project_manifest():
    manifest = ProjectManifest(project_name="Test Project", dataset_id="123")
    assert manifest.quality_preset.inference_steps == 20

def test_story_bible():
    char = CharacterReference(name="Alice", visual_dna="tall", outfit="dress", color_palette="red")
    bible = StoryBible(characters={"alice": char})
    assert bible.version == 1
    assert "alice" in bible.characters
