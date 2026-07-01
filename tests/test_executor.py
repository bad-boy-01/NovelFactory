import pytest
from core.pipeline.context import PipelineContext
from core.pipeline.executor import SequentialExecutor
from core.pipeline.stage import PipelineStage
from core.domain.project import ProjectManifest

class MockStageA:
    def get_name(self) -> str: return "Stage A"
    def fingerprint(self, context: PipelineContext) -> str: return "hashA"
    def execute(self, context: PipelineContext) -> PipelineContext:
        context.state["A"] = True
        return context

class MockStageB:
    def get_name(self) -> str: return "Stage B"
    def fingerprint(self, context: PipelineContext) -> str: return "hashB"
    def execute(self, context: PipelineContext) -> PipelineContext:
        context.state["B"] = True
        return context

def test_sequential_executor():
    manifest = ProjectManifest(project_name="Test", dataset_id="Dataset")
    ctx = PipelineContext(project_manifest=manifest)
    
    executor = SequentialExecutor(stages=[MockStageA(), MockStageB()])
    final_ctx = executor.run(ctx)
    
    assert final_ctx.state.get("A") is True
    assert final_ctx.state.get("B") is True
