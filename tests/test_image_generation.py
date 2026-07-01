import pytest
from pathlib import Path
from core.pipeline.context import PipelineContext
from core.domain.project import ProjectManifest
from core.domain.story import Scene, Beat
from core.domain.prompt import DeclarativePrompt
from core.domain.bible import StoryBible
from ai.generation.image import ImageGenerationStage

class MockImageGeneratorProvider:
    def get_model_name(self) -> str: return "MockFLUX"
    def get_model_revision(self) -> str: return "v1.0"
    def generate_image(self, prompt: str, negative_prompt: str, seed: int, output_path: Path) -> Path:
        output_path.write_text("fake png data")
        return output_path

def test_image_generation_stage(tmp_path):
    manifest = ProjectManifest(project_name="Test", dataset_id="Dataset")
    ctx = PipelineContext(project_manifest=manifest)
    ctx.story_bible = StoryBible(hash="xyz")
    
    beat = Beat(text="He smiled")
    ctx.current_scene = Scene(beats=[beat])
    
    dp = DeclarativePrompt(compiled_text="masterpiece, 1boy, smiling")
    ctx.prompts[beat.id] = dp
    
    out_dir = tmp_path / "outputs"
    provider = MockImageGeneratorProvider()
    stage = ImageGenerationStage(provider=provider, output_dir=out_dir)
    
    assert stage.get_name() == "Image Generator (MockFLUX)"
    
    ctx = stage.execute(ctx)
    
    # Verify Asset Creation
    assert len(ctx.assets) == 1
    asset = list(ctx.assets.values())[0]
    
    # Verify Provenance
    assert asset.provenance.model == "MockFLUX"
    assert asset.provenance.model_revision == "v1.0"
    assert asset.provenance.seed == 42
    assert asset.provenance.story_bible_hash == "xyz"
    assert asset.provenance.generated_from == dp.id
    
    # Verify File
    assert asset.file_path.exists()
    assert asset.file_path.read_text() == "fake png data"
