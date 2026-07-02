import logging
from typing import List
from core.pipeline.context import PipelineContext
from core.pipeline.stage import PipelineStage
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
        
        from core.pipeline.reducer import ContextReducer
        self.reducer = ContextReducer()

    def run(self, context: PipelineContext) -> PipelineContext:
        import time
        logger.info("Starting Sequential Executor")
        
        current_context = context
        timeline_logs = []
        pipeline_start_time = time.time()
        
        for i, stage in enumerate(self.stages):
            retries = 0
            stage_name = stage.__class__.__name__
            stage_start_time = time.time()

            # Lifecycle: LOAD
            for provider in stage.get_providers():
                if hasattr(provider, 'load'):
                    provider.load()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:
                pass

            while True:
                logger.info(f"Starting stage: {stage_name} (Attempt {retries + 1})")
                
                try:
                    candidate_result = stage.execute(current_context)
                    
                    contracts = self.router.get_contracts(stage_name)
                    engine = ContractEngine(contracts)
                    engine.run(current_context, candidate_result.artifact)
                    
                    current_context = self.reducer.reduce(current_context, candidate_result)
                    break  # PASS
                except Exception as e:
                    retries += 1
                    logger.warning(f"[RETRY] {stage_name}: Failed ({retries}/{self.max_retries}) - {str(e)}", exc_info=True)
                    if retries >= self.max_retries:
                        logger.error(f"[FATAL] {stage_name} exhausted all retries.", exc_info=True)
                        raise e

            # Lifecycle: UNLOAD
            for provider in stage.get_providers():
                if hasattr(provider, 'unload'):
                    provider.unload()

            stage_duration = time.time() - stage_start_time
            peak_vram = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            except ImportError:
                pass
            
            cache_hit = candidate_result.metrics.get('cache_hit', False) if candidate_result else False
            cache_str = "HIT" if cache_hit else "MISS"
            
            timeline_logs.append(
                f"[{i+1}/{len(self.stages)}] {stage_name:<20} {stage_duration:.2f} s\n"
                f"        Cache: {cache_str}\n"
                f"        Peak VRAM: {peak_vram:.2f} GB\n"
                f"        Contracts: PASS\n"
                f"        Retry: {retries}"
            )

        logger.info("\n" + "="*40 + "\nSTAGE TIMELINE\n" + "="*40)
        for log in timeline_logs:
            logger.info(log)
            
        total_time = time.time() - pipeline_start_time
        logger.info(f"\nTotal Time: {total_time:.2f} s")
        logger.info("="*40 + "\n")
        
        return current_context
