from core.pipeline.stage import StageResult
from core.domain.asset import ExecutionNode
from core.prompt.ast import PromptAST

class PromptBuilderStage:
    def __init__(self):
        pass
        
    def get_providers(self) -> list:
        return []

    def execute(self, context) -> StageResult:
        bible = context.story_bible
        
        if not bible or not bible.characters:
            raise ValueError("StoryBible missing or has no characters.")
            
        char = list(bible.characters.values())[0]
        
        ast = PromptAST(
            character=char.name,
            outfit=char.outfit,
            scene="dark forest illuminated by moonlight",
            camera="medium shot, centered",
            lighting="soft cinematic lighting",
            style="anime cinematic, high detail",
            negative="low quality, blurry, distorted"
        )
        
        node = ExecutionNode(artifact=ast, stage_name="PromptBuilderStage")
        
        return StageResult(
            artifact=ast,
            execution_node=node,
            metrics={},
            metadata={}
        )
