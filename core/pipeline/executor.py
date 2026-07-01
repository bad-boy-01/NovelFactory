import logging
from typing import List
from .context import PipelineContext
from .stage import PipelineStage
from core.contracts.engine import ContractEngine
from core.contracts.router import ContractRouter

logger = logging.getLogger(__name__)

class SequentialExecutor:
    """
    Executor that runs stages sequentially, guarded by the Generative Contract System.
    """
    def __init__(self, stages: List[PipelineStage], contract_router: ContractRouter, max_retries: int = 2):
        self.stages = stages
        self.router = contract_router
        self.max_retries = max_retries

    def run(self, context: PipelineContext) -> PipelineContext:
        logger.info("Starting Sequential Executor")
        
        for stage in self.stages:
            retries = 0
            stage_name = stage.__class__.__name__

            while True:
                logger.info(f"Starting stage: {stage_name} (Attempt {retries + 1})")
                
                # 1. Execute stage
                artifact = stage.execute(context)

                # 2. HARD CONTRACT CHECK (NEW CORE LOGIC)
                contracts = self.router.get_contracts(stage_name)
                engine = ContractEngine(contracts)

                try:
                    engine.run(context, artifact)
                    # 3. Commit result ONLY after validation
                    # Note: We append artifact if your stages return them, otherwise stages mutate context directly.
                    # Currently, stage.execute(context) mutates and returns context, so artifact here is context.
                    # Let's assume stage execution mutates context in a replay-safe way.
                    break  # PASS → continue pipeline

                except Exception as e:
                    retries += 1
                    logger.warning(f"[RETRY] {stage_name}: Failed contract validation ({retries}/{self.max_retries}) - {str(e)}")

                    if retries > self.max_retries:
                        logger.error(f"[FATAL] {stage_name} exhausted all retries.")
                        raise e
            
            logger.info(f"Completed stage: {stage_name}")

        logger.info("Pipeline execution complete")
        return context
