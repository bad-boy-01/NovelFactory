from typing import Any, Dict, List, Protocol
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class RenderArtifact(BaseModel):
    kind: str # e.g. "PROMPT", "IMAGE", "UPSCALE"
    data: Any

class RenderNode(Protocol):
    def get_name(self) -> str:
        ...
        
    def execute(self, inputs: Dict[str, RenderArtifact], config: Dict[str, Any]) -> Dict[str, RenderArtifact]:
        ...

class RenderGraph:
    def __init__(self):
        self.nodes: List[RenderNode] = []
        
    def add_node(self, node: RenderNode):
        self.nodes.append(node)
        
    def execute(self, initial_artifacts: Dict[str, RenderArtifact], config: Dict[str, Any]) -> Dict[str, RenderArtifact]:
        current_state = dict(initial_artifacts)
        for node in self.nodes:
            logger.info(f"Executing RenderNode: {node.get_name()}")
            try:
                outputs = node.execute(current_state, config)
                current_state.update(outputs)
            except Exception as e:
                logger.error(f"RenderNode {node.get_name()} failed: {e}")
                raise e
        return current_state
