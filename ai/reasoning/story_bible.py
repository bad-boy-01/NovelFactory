from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext
from core.domain.bible import StoryBible, CharacterReference
from plugins.interfaces import LLMProvider

class StoryBibleGeneratorStage(PipelineStage):
    """
    Stage that analyzes the initial novel text and extracts global 
    continuity rules, such as character visual definitions and outfits.
    """
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def get_name(self) -> str:
        return "Story Bible Generator"

    def fingerprint(self, context: PipelineContext) -> str:
        return f"bible_{context.project_manifest.dataset_id}"

    def execute(self, context: PipelineContext) -> PipelineContext:
        # If the context already has a story bible, skip generation.
        # This allows resuming from checkpoints.
        if context.story_bible:
            return context
            
        prompt = "Extract main characters and return JSON with name, visual_dna, outfit, color_palette."
        
        try:
            response = self.llm.generate_json(
                prompt=prompt, 
                system_prompt="You are a story analyst. Output strictly valid JSON."
            )
            
            characters = {}
            for char_data in response.get("characters", []):
                char = CharacterReference(
                    name=char_data.get("name", "Unknown"),
                    visual_dna=char_data.get("visual_dna", ""),
                    outfit=char_data.get("outfit", ""),
                    color_palette=char_data.get("color_palette", "")
                )
                characters[char.name.lower()] = char
                
            context.story_bible = StoryBible(
                version=1, 
                characters=characters, 
                generator_model="llm_provider"
            )
        except Exception as e:
            # Fallback to empty bible if LLM fails
            context.story_bible = StoryBible(version=1, generator_model="error_fallback")
            
        return context
