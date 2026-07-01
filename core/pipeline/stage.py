from typing import Protocol, TypeVar, Generic, Mapping, Any
from dataclasses import dataclass
from .context import PipelineContext

T = TypeVar("T")

@dataclass(frozen=True)
class StageResult(Generic[T]):
    artifact: T
    execution_node: Any  # core.domain.asset.ExecutionNode
    metrics: Mapping[str, Any]
    metadata: Mapping[str, Any]


class PipelineStage(Protocol):
    """
    A single stage in the AI pipeline.
    Must implement execute() to return a StageResult.
    """
    
    def get_name(self) -> str:
        """Returns the human-readable name of the stage."""
        ...
        
    def get_providers(self) -> list:
        """Returns a list of providers used by this stage, so the executor can manage their lifecycle."""
        return []
        
    def fingerprint(self, context: PipelineContext) -> str:
        """
        Returns a deterministic hash based on inputs, config, and model versions.
        Used by the executor to skip unchanged stages via caching.
        """
        ...
        
    def execute(self, context: PipelineContext) -> StageResult:
        """
        Executes the stage logic without mutating the context directly,
        returning a StageResult that the executor will later commit.
        """
        ...
