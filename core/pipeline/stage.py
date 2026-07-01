from typing import Protocol
from .context import PipelineContext

class PipelineStage(Protocol):
    """
    A single stage in the AI pipeline.
    Must implement execute() to mutate the context.
    """
    
    def get_name(self) -> str:
        """Returns the human-readable name of the stage."""
        ...
        
    def fingerprint(self, context: PipelineContext) -> str:
        """
        Returns a deterministic hash based on inputs, config, and model versions.
        Used by the executor to skip unchanged stages via caching.
        """
        ...
        
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Executes the stage logic, mutating and returning the PipelineContext.
        """
        ...
