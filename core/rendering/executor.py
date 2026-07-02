import logging
from typing import List
from core.pipeline.context import PipelineContext
from core.pipeline.stage import PipelineStage
from core.contracts.engine import ContractEngine
from core.contracts.router import ContractRouter

logger = logging.getLogger(__name__)

class CompilerExecutor:
    """
    Executor that runs stages, guarded by the Generative Contract System and Incremental Cache.
    """
    def __init__(self, stages: List[PipelineStage], contract_router: ContractRouter, max_retries: int = 2):
        self.stages = stages
        self.router = contract_router
        self.max_retries = max_retries
        
        from core.pipeline.reducer import ContextReducer
        self.reducer = ContextReducer()

    def run(self, context: PipelineContext) -> PipelineContext:
        import time
        logger.info("Starting Compiler Executor")
        
        current_context = context
        timeline_logs = []
        pipeline_start_time = time.time()
        
        for i, stage in enumerate(self.stages):
            retries = 0
            stage_name = stage.__class__.__name__
            stage_start_time = time.time()

            # Note: Provider loading/unloading is now managed externally by ResourceSession.

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:
                pass

            # Incremental Cache Check
            cache_hit = False
            candidate_result = None
            if hasattr(stage, "get_dependency_hash") and hasattr(stage, "load_cached_artifact"):
                current_dep_hash = stage.get_dependency_hash(current_context)
                if hasattr(current_context, "workspace"):
                    cached_artifact = stage.load_cached_artifact(current_context.workspace)
                    if cached_artifact:
                        if getattr(cached_artifact, "dependency_hash", None) == current_dep_hash:
                            logger.info(f"[{stage_name}] Cache HIT (Dependency hash matched). Skipping execution.")
                            cache_hit = True
                            
                            from core.pipeline.stage import StageResult
                            from core.domain.assets.execution import ExecutionNode
                            node = ExecutionNode(artifact=cached_artifact, stage_name=stage_name)
                            candidate_result = StageResult(
                                artifact=cached_artifact,
                                execution_node=node,
                                metrics={"cache_hit": True},
                                metadata={}
                            )
                        else:
                            logger.info(f"[{stage_name}] Cache MISS (Dependency hash changed).")

            while not cache_hit:
                logger.info(f"Starting stage: {stage_name} (Attempt {retries + 1})")
                
                try:
                    candidate_result = stage.execute(current_context)
                    candidate_result.metrics["cache_hit"] = False
                    
                    contracts = self.router.get_contracts(stage_name)
                    engine = ContractEngine(contracts)
                    engine.run(current_context, candidate_result.artifact)
                    break  # PASS
                except Exception as e:
                    retries += 1
                    logger.warning(f"[RETRY] {stage_name}: Failed ({retries}/{self.max_retries}) - {str(e)}", exc_info=True)
                    if retries >= self.max_retries:
                        logger.error(f"[FATAL] {stage_name} exhausted all retries.", exc_info=True)
                        raise e
            
            # Reduce context whether from cache or fresh execution
            if candidate_result:
                current_context = self.reducer.reduce(current_context, candidate_result)

            stage_duration = time.time() - stage_start_time
            peak_alloc = 0.0
            peak_res = 0.0
            cur_alloc = 0.0
            cur_res = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
                    peak_res = torch.cuda.max_memory_reserved() / (1024**3)
                    cur_alloc = torch.cuda.memory_allocated() / (1024**3)
                    cur_res = torch.cuda.memory_reserved() / (1024**3)
            except ImportError:
                pass
            
            cache_hit = candidate_result.metrics.get('cache_hit', False) if candidate_result else False
            cache_str = "HIT" if cache_hit else "MISS"
            
            timeline_logs.append(
                f"[{i+1}/{len(self.stages)}] {stage_name:<20} {stage_duration:.2f} s\n"
                f"        Cache: {cache_str}\n"
                f"        Alloc: {cur_alloc:.2f} GB (Peak: {peak_alloc:.2f} GB)\n"
                f"        Reserv: {cur_res:.2f} GB (Peak: {peak_res:.2f} GB)\n"
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
