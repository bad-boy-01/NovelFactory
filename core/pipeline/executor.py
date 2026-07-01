import logging
from typing import List
from .context import PipelineContext
from .stage import PipelineStage

logger = logging.getLogger(__name__)

class SequentialExecutor:
    """
    MVP executor that runs a static list of stages sequentially.
    Later, this will evolve into the HybridExecutor.
    """
    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages
        
    def run(self, context: PipelineContext) -> PipelineContext:
        logger.info("Starting Sequential Executor")
        for stage in self.stages:
            stage_name = stage.get_name()
            logger.info(f"Starting stage: {stage_name}")
            
            # TODO: Future versions will check stage.fingerprint() against the SceneCache 
            # to skip execution if inputs haven't changed.
            
            context = stage.execute(context)
            
            logger.info(f"Completed stage: {stage_name}")
            # TODO: Checkpoint logic would intercept here.
            
        logger.info("Pipeline execution complete")
        return context
