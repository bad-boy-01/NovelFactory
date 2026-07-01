import pytest
from core.pipeline.context import PipelineContext
from core.domain.project import ProjectManifest
from core.domain.story import Scene, Beat, VisualScreenplay
from core.domain.bible import StoryBible, CharacterReference
from core.profiles.prompt_pack import PromptPack
from ai.prompting.compiler import PromptCompilerStage

def test_prompt_compiler():
    manifest = ProjectManifest(project_name="Test", dataset_id="Dataset")
    ctx = PipelineContext(project_manifest=manifest)
    
    # Setup Story Bible
    char = CharacterReference(name="Alice", visual_dna="1girl, blonde", outfit="red dress", color_palette="red")
    ctx.story_bible = StoryBible(characters={"alice": char})
    
    # Setup Scene with Visual Screenplay
    vs = VisualScreenplay(
        characters=["Alice"],
        emotion="happy",
        camera="close-up",
        lighting="sunset",
        mood="garden"
    )
    beat = Beat(text="She smiled.", visual_screenplay=vs)
    ctx.current_scene = Scene(beats=[beat])
    
    # Setup Prompt Pack
    pack = PromptPack(name="Anime", positive_prefix="masterpiece anime style", positive_suffix="highly detailed")
    
    # Execute Compiler
    stage = PromptCompilerStage(prompt_pack=pack)
    ctx = stage.execute(ctx)
    
    # Verify Context State
    assert beat.id in ctx.prompts
    prompt = ctx.prompts[beat.id]
    
    # Verify AST
    assert prompt.style == "Anime"
    assert "Alice" in prompt.characters
    assert prompt.camera == "close-up"
    
    # Verify Compiled Text
    assert "masterpiece anime style" in prompt.compiled_text
    assert "1girl, blonde, wearing red dress" in prompt.compiled_text
    assert "Expression: happy" in prompt.compiled_text
    assert "close-up" in prompt.compiled_text
    assert "sunset" in prompt.compiled_text
    assert "highly detailed" in prompt.compiled_text
