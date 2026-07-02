from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.prompt.ast import PromptManifest
from core.domain.assets.registry import AssetRegistry, Asset
from core.rendering.render_queue import RenderQueue
import logging
import os
import json
import math
import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CompilePromptNode(RenderNode):
    def __init__(self, provider_compiler):
        self.compiler = provider_compiler
        
    def get_name(self) -> str:
        return "CompilePromptNode"
        
    def execute(self, inputs: Dict[str, RenderArtifact], config: Dict[str, Any]) -> Dict[str, RenderArtifact]:
        plan = inputs["RENDER_PLAN"].data
        request = self.compiler.compile_plan(plan)
        return {"PROVIDER_REQUEST": RenderArtifact(kind="PROVIDER_REQUEST", data=request)}

class GenerateNode(RenderNode):
    def __init__(self, provider):
        self.provider = provider
        
    def get_name(self) -> str:
        return "GenerateNode"
        
    def execute(self, inputs: Dict[str, RenderArtifact], config: Dict[str, Any]) -> Dict[str, RenderArtifact]:
        request = inputs["PROVIDER_REQUEST"].data
        image = self.provider.generate(request)
        return {"IMAGE": RenderArtifact(kind="IMAGE", data=image)}

class SaveNode(RenderNode):
    def get_name(self) -> str:
        return "SaveNode"
        
    def execute(self, inputs: Dict[str, RenderArtifact], config: Dict[str, Any]) -> Dict[str, RenderArtifact]:
        image = inputs["IMAGE"].data
        shot_dir = config["shot_dir"]
        output_path = str(shot_dir / "image.png")
        image.save(output_path)
        return {"SAVED_IMAGE_PATH": RenderArtifact(kind="PATH", data=output_path)}

class DiffusionRendererStage(PipelineStage):
    def __init__(self, provider_compiler=None, diffusion_provider=None, render_options=None):
        self.compiler = provider_compiler
        self.diffusion = diffusion_provider
        self.render_options = render_options or {}

    def get_providers(self) -> list:
        return [self.diffusion] if self.diffusion else []

    def execute(self, context) -> StageResult:
        if not self.compiler:
            from plugins.local_diffusion import DiffusersCompiler
            self.compiler = DiffusersCompiler()
            
        if not self.diffusion:
            from plugins.local_diffusion import DiffusersProvider
            self.diffusion = DiffusersProvider()
            
        # In the real pipeline, we'd extract a list of RenderPlans.
        # But we don't have them connected yet, so we just log the action.
        logger.info("Executing DiffusionRendererStage via RenderGraph...")
        
        # Build the graph
        graph = RenderGraph()
        graph.add_node(CompilePromptNode(self.compiler))
        graph.add_node(GenerateNode(self.diffusion))
        graph.add_node(SaveNode())
        
        # We would loop over RenderPlans here
        # For now, just return empty registry to preserve interface
        registry = context.registry
        node = ExecutionNode(artifact=registry, stage_name="DiffusionRendererStage")
        
        return StageResult(
            artifact=registry,
            execution_node=node,
            metrics={"jobs_processed": 0},
            metadata={}
        )
