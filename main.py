import argparse
import logging
import json
from pathlib import Path
import shutil

from core.pipeline.context import PipelineContext
from core.domain.project import ProjectManifest
from plugins.local_llm import LocalLLMProvider
from ai.reasoning.story_bible import StoryBibleGeneratorStage
from plugins.local_diffusion import DiffusionProvider
from plugins.interfaces import DiffusionConfig
from ai.generation.image_stage import ImageGenerationStage
from core.pipeline.cache import CacheProvider
from ai.prompting.prompt_stage import PromptBuilderStage
from plugins.ffmpeg_renderer import FFmpegVideoRenderer
from ai.generation.rendering_stage import RenderingStage
from core.pipeline.executor import SequentialExecutor
from core.contracts.router import ContractRouter

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("NovelFactory")

def save_workspace(name: str, content: str):
    Path("workspace").mkdir(exist_ok=True)
    with open(f"workspace/{name}", "w", encoding="utf-8") as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description="NovelFactory Milestone 1 Validator")
    parser.add_argument("--novel", type=str, default="sample.txt", help="Path to novel text")
    parser.add_argument("--stage", type=str, default="all", help="Stage to run: story_bible, prompt, diffusion, render, all, validate")
    args = parser.parse_args()

    if args.stage == "validate":
        logger.info("[VALIDATE] Running architectural sanity check...")
        manifest = ProjectManifest(project_name="Sanity", dataset_id="test", source_text="test")
        context = PipelineContext(project_manifest=manifest)
        llm = LocalLLMProvider()
        cache = CacheProvider(".cache")
        diff = DiffusionProvider(DiffusionConfig(cpu_offload=True))
        rend = FFmpegVideoRenderer()
        stages = [StoryBibleGeneratorStage(llm), PromptBuilderStage(), ImageGenerationStage(diff, cache), RenderingStage(rend)]
        router = ContractRouter()
        from core.pipeline.reducer import ContextReducer
        from core.pipeline.stage import StageResult
        reducer = ContextReducer()
        dummy_result = StageResult(artifact="dummy", metrics={}, metadata={})
        new_ctx = reducer.reduce(context, dummy_result)
        assert new_ctx is not context, "Reducer mutated context instead of copying!"
        logger.info("[SUCCESS] Sanity validation passed. All architecture components wired correctly.")
        return

    Path("workspace").mkdir(exist_ok=True)
    Path("debug").mkdir(exist_ok=True)

    if not Path(args.novel).exists():
        with open(args.novel, "w") as f:
            f.write("Alice stood in the dark forest, her blue dress illuminated by moonlight.")

    with open(args.novel, "r", encoding="utf-8") as f:
        novel_text = f.read()

    manifest = ProjectManifest(project_name="Milestone1", dataset_id="local_run", source_text=novel_text)
    context = PipelineContext(project_manifest=manifest)

    llm_provider = LocalLLMProvider() 
    cache_provider = CacheProvider(cache_dir=".cache/generation")
    diff_config = DiffusionConfig(cpu_offload=True) 
    diffusion_provider = DiffusionProvider(config=diff_config)
    renderer_provider = FFmpegVideoRenderer()

    # Stages need to expose get_providers for Executor lifecycle management
    sb_stage = StoryBibleGeneratorStage(llm_provider)
    setattr(sb_stage, 'get_providers', lambda: [llm_provider])
    
    prompt_stage = PromptBuilderStage()
    setattr(prompt_stage, 'get_providers', lambda: [])
    
    img_stage = ImageGenerationStage(provider=diffusion_provider, cache=cache_provider)
    setattr(img_stage, 'get_providers', lambda: [diffusion_provider])
    
    render_stage = RenderingStage(renderer=renderer_provider, output_file="workspace/008_video.mp4")
    setattr(render_stage, 'get_providers', lambda: [renderer_provider])

    stages_map = {
        "story_bible": sb_stage,
        "prompt": prompt_stage,
        "diffusion": img_stage,
        "render": render_stage
    }

    if args.stage == "all":
        active_stages = list(stages_map.values())
    else:
        active_stages = [stages_map[args.stage]]

    router = ContractRouter()
    executor = SequentialExecutor(stages=active_stages, contract_router=router, max_retries=2)

    try:
        final_context = executor.run(context)
        
        # Save intermediate artifacts
        if final_context.story_bible:
            save_workspace("001_story_bible.json", final_context.story_bible.model_dump_json(indent=2))
        
        # Copy workspace to debug
        shutil.copytree("workspace", "debug/workspace_run", dirs_exist_ok=True)

        logger.info("\n[SUCCESS] Pipeline execution complete.")
    except Exception as e:
        logger.error(f"\n[ERROR] Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()
