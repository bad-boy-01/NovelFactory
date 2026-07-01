from core.pipeline.context import PipelineContext
from core.pipeline.stage import StageResult

class ContextReducer:
    """
    Pure Redux-style reducer that takes a frozen PipelineContext and a validated StageResult,
    and returns a brand new PipelineContext (Context N+1).
    Does not run validations or retries.
    """
    def reduce(self, context: PipelineContext, result: StageResult) -> PipelineContext:
        # Create a dictionary of current context values
        context_dict = context.model_dump()
        
        # Determine how to merge the artifact based on its type or metadata
        # For MVP, we naively append to assets if it's an asset, or update story_bible, etc.
        # Ideally, we rely on result.metadata or explicit artifact types.
        
        from core.domain.asset import Asset
        from core.domain.bible import StoryBible
        
        if isinstance(result.artifact, StoryBible):
            context_dict['story_bible'] = result.artifact
            
        elif isinstance(result.artifact, Asset):
            # context.assets is a dict of UUID to Asset
            import uuid
            # For simplicity, generate a new UUID or use asset_id
            asset_id = getattr(result.artifact, 'asset_id', str(uuid.uuid4()))
            
            # Since context is frozen, we must copy the dict
            new_assets = dict(context_dict.get('assets', {}))
            new_assets[asset_id] = result.artifact
            context_dict['assets'] = new_assets
            
        else:
            # If the stage produced something else, we might just store it in ephemeral state
            new_state = dict(context_dict.get('state', {}))
            new_state[str(type(result.artifact).__name__)] = result.artifact
            context_dict['state'] = new_state
            
        return PipelineContext(**context_dict)
