from core.pipeline.stage import StageResult
from core.domain.asset import ExecutionNode
from core.domain.bible import StoryBible, CharacterReference

class StoryBibleGeneratorStage:
    def __init__(self, llm_provider):
        self.llm = llm_provider

    def get_providers(self) -> list:
        return [self.llm]

    def execute(self, context) -> StageResult:
        schema = {
            "characters": [
                {
                    "name": "string",
                    "appearance": "string",
                    "outfit": "string"
                }
            ]
        }
        
        raw_text = context.project_manifest.source_text
        prompt = f"Extract structured character data from this story:\n{raw_text}"
        
        result_dict = self.llm.generate_json(prompt, schema)
        
        chars = [CharacterReference(**c) for c in result_dict.get("characters", [])]
        bible = StoryBible(characters=chars)
        
        node = ExecutionNode(artifact=bible, stage_name="StoryBibleGeneratorStage")
        
        return StageResult(
            artifact=bible,
            execution_node=node,
            metrics={},
            metadata={}
        )
