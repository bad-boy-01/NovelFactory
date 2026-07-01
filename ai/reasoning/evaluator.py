from typing import List
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext
from core.domain.asset import AssetType, EvaluationResult
from plugins.interfaces import EvaluatorPlugin

class EvaluationStage(PipelineStage):
    """
    Runs a chain of EvaluatorPlugins across generated assets.
    Aggregates scores and flags assets that require regeneration.
    """
    def __init__(self, evaluators: List[EvaluatorPlugin], threshold: float = 0.8):
        self.evaluators = evaluators
        self.threshold = threshold

    def get_name(self) -> str:
        return "Evaluation Chain"

    def fingerprint(self, context: PipelineContext) -> str:
        return "eval_" + "_".join([e.get_name() for e in self.evaluators])

    def execute(self, context: PipelineContext) -> PipelineContext:
        for asset in context.assets.values():
            if asset.asset_type != AssetType.IMAGE:
                continue
                
            if not self.evaluators:
                continue
                
            total_score = 0.0
            reasons = []
            retry = False
            
            for evaluator in self.evaluators:
                result = evaluator.evaluate(asset, context)
                total_score += result.score
                if result.reason:
                    reasons.append(f"{evaluator.get_name()}: {result.reason}")
                if result.retry_needed:
                    retry = True
                    
            avg_score = total_score / len(self.evaluators)
            
            asset.evaluation = EvaluationResult(
                score=avg_score,
                reason=" | ".join(reasons),
                retry_needed=retry or avg_score < self.threshold
            )
            
        return context
