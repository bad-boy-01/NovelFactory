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
                    "visual_dna": "string",
                    "outfit": "string",
                    "color_palette": "string"
                }
            ]
        }
        
        raw_text = context.project_manifest.source_text
        # Safety limit for Milestone 1: Truncate massive novels to prevent OOM during generation
        if len(raw_text) > 10000:
            raw_text = raw_text[:10000] + "\n...[TRUNCATED]"
            
        prompt = f"Extract structured character data from this story:\n{raw_text}"
        
        result_dict = self.llm.generate_json(prompt, schema)
        
        chars = {c["name"]: CharacterReference(**c) for c in result_dict.get("characters", [])}
        bible = StoryBible(characters=chars)
        
        node = ExecutionNode(artifact=bible, stage_name="StoryBibleGeneratorStage")
        
        return StageResult(
            artifact=bible,
            execution_node=node,
            metrics={},
            metadata={}
        )
