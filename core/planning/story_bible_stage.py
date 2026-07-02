from core.pipeline.stage import StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.story.bible import StoryBible, CharacterReference

class StoryBibleGeneratorStage:
    def __init__(self, llm_provider):
        self.llm = llm_provider

    def get_providers(self) -> list:
        return [self.llm]

    def get_dependency_hash(self, context) -> str:
        import hashlib
        return hashlib.sha256(context.project_manifest.source_text.encode("utf-8")).hexdigest()
        
    def load_cached_artifact(self, workspace):
        import json
        cached = workspace.load_json("manifests/story_bible.json")
        if cached:
            return StoryBible.model_validate(cached)
        return None

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
        if len(raw_text) > 10000:
            raw_text = raw_text[:10000] + "\n...[TRUNCATED]"
            
        prompt = f"Extract structured character data from this story:\n{raw_text}"
        
        result_dict = self.llm.generate_json(prompt, schema)
        
        chars = {c["name"]: CharacterReference(**c) for c in result_dict.get("characters", [])}
        
        # Incremental compilation metadata
        import hashlib
        dep_hash = self.get_dependency_hash(context)
        bible = StoryBible(
            characters=chars,
            dependency_hash=dep_hash,
            generator="StoryBibleGeneratorStage",
            pipeline_version="0.4.4"
        )
        # Compute own fingerprint
        bible.fingerprint = hashlib.sha256(bible.model_dump_json(exclude={"fingerprint", "created_at"}).encode()).hexdigest()
        
        # Save to Workspace
        if hasattr(context, "workspace"):
            context.workspace.save_json("manifests/story_bible.json", bible.model_dump(mode="json"))
        
        node = ExecutionNode(artifact=bible, stage_name="StoryBibleGeneratorStage")
        
        return StageResult(
            artifact=bible,
            execution_node=node,
            metrics={},
            metadata={}
        )
