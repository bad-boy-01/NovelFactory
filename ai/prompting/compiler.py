from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext
from core.domain.prompt import DeclarativePrompt
from core.profiles.prompt_pack import PromptPack

class PromptCompilerStage(PipelineStage):
    """
    Compiles the VisualScreenplay and StoryBible into a final 
    DeclarativePrompt AST, then generates the compiled_text 
    based on the loaded PromptPack.
    """
    def __init__(self, prompt_pack: PromptPack):
        self.prompt_pack = prompt_pack
        
    def get_name(self) -> str:
        return f"Prompt Compiler ({self.prompt_pack.name})"
        
    def fingerprint(self, context: PipelineContext) -> str:
        scene_id = context.current_scene.id if context.current_scene else "none"
        bible_hash = context.story_bible.hash if context.story_bible else "none"
        return f"compile_{scene_id}_{bible_hash}_{self.prompt_pack.name}"

    def execute(self, context: PipelineContext) -> PipelineContext:
        if not context.current_scene:
            return context
            
        for beat in context.current_scene.beats:
            vs = beat.visual_screenplay
            
            # 1. Resolve Characters against Story Bible
            char_visuals = {}
            if context.story_bible:
                for char_name in vs.characters:
                    ref = context.story_bible.characters.get(char_name.lower())
                    if ref:
                        # Use Bible outfit as fallback if scene directives dictate, etc.
                        char_visuals[char_name] = f"{ref.visual_dna}, wearing {ref.outfit}"
            
            # 2. Construct AST
            dp = DeclarativePrompt(
                characters=char_visuals,
                expression=vs.emotion,
                camera=vs.camera,
                lighting=vs.lighting,
                environment=vs.mood,
                style=self.prompt_pack.name
            )
            
            # 3. Compile to string
            char_str = ", ".join(char_visuals.values())
            components = [
                self.prompt_pack.positive_prefix,
                char_str,
                f"Expression: {dp.expression}" if dp.expression else "",
                dp.camera,
                dp.lighting,
                dp.environment,
                self.prompt_pack.positive_suffix
            ]
            
            # Filter empty strings and join
            compiled = ", ".join(c for c in components if c.strip())
            dp.compiled_text = compiled
            
            context.prompts[beat.id] = dp
            
        return context
