import pytest
from pathlib import Path
from core.pipeline.context import PipelineContext
from core.domain.project import ProjectManifest
from core.domain.asset import Asset, AssetType, EvaluationResult
from core.domain.base import ProvenanceGraph
from ai.reasoning.evaluator import EvaluationStage

class MockCharacterEvaluator:
    def get_name(self) -> str: return "CharacterEval"
    def evaluate(self, asset: Asset, context: PipelineContext) -> EvaluationResult:
        # Pretend the character is mostly correct
        return EvaluationResult(score=0.9, reason="", retry_needed=False)

class MockNSFWEvaluator:
    def get_name(self) -> str: return "NSFWEval"
    def evaluate(self, asset: Asset, context: PipelineContext) -> EvaluationResult:
        # Pretend there is a minor text artifact but not a total failure
        return EvaluationResult(score=0.6, reason="Minor text artifacts", retry_needed=True)

def test_evaluation_stage():
    manifest = ProjectManifest(project_name="Test", dataset_id="Dataset")
    ctx = PipelineContext(project_manifest=manifest)
    
    # Add an image asset to context
    asset = Asset(
        asset_type=AssetType.IMAGE,
        file_path=Path("/tmp/img.png"),
        hash_sha256="abc",
        provenance=ProvenanceGraph()
    )
    ctx.assets[asset.id] = asset
    
    stage = EvaluationStage(evaluators=[MockCharacterEvaluator(), MockNSFWEvaluator()], threshold=0.8)
    assert stage.get_name() == "Evaluation Chain"
    
    ctx = stage.execute(ctx)
    
    evaluated_asset = ctx.assets[asset.id]
    assert evaluated_asset.evaluation is not None
    assert evaluated_asset.evaluation.score == 0.75 # (0.9 + 0.6) / 2
    assert evaluated_asset.evaluation.retry_needed is True
    assert "NSFWEval: Minor text artifacts" in evaluated_asset.evaluation.reason
