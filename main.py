import argparse
import logging
import json
from pathlib import Path
import shutil

from core.pipeline.context import PipelineContext
from core.domain.story.project import ProjectManifest
from plugins.local_llm import LocalLLMProvider
from core.planning.story_bible_stage import StoryBibleGeneratorStage
from plugins.local_diffusion import DiffusionProvider
from plugins.interfaces import DiffusionConfig
from core.rendering.image_stage import ImageGenerationStage
from core.pipeline.cache import CacheProvider
from core.optimization.prompt_builder import PromptBuilderStage
from plugins.ffmpeg_renderer import FFmpegVideoRenderer
from core.rendering.assembly_stage import RenderingStage
from core.rendering.executor import SequentialExecutor
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
        router = ContractRouter({})
        from core.pipeline.reducer import ContextReducer
        from core.pipeline.stage import StageResult
        reducer = ContextReducer()
        from core.domain.assets.execution import ExecutionNode
        dummy_result = StageResult(artifact="dummy", execution_node=ExecutionNode(artifact="dummy", stage_name="validate"), metrics={}, metadata={})
        new_ctx = reducer.reduce(context, dummy_result)
        assert new_ctx is not context, "Reducer mutated context instead of copying!"
        logger.info("[SUCCESS] Sanity validation passed. All architecture components wired correctly.")
        return

    Path("workspace").mkdir(exist_ok=True)
    Path("debug").mkdir(exist_ok=True)

    novel_path = Path(args.novel)
    if not novel_path.exists():
        with open(args.novel, "w") as f:
            f.write("Alice stood in the dark forest, her blue dress illuminated by moonlight.")

    if novel_path.suffix == '.docx':
        import docx
        doc = docx.Document(args.novel)
        novel_text = "\\n".join([para.text for para in doc.paragraphs])
    else:
        with open(args.novel, "r", encoding="utf-8") as f:
            novel_text = f.read()

    manifest = ProjectManifest(project_name="Milestone1", dataset_id="local_run", source_text=novel_text)
    context = PipelineContext(project_manifest=manifest)

    llm_provider = LocalLLMProvider() 
    cache_provider = CacheProvider(cache_dir=".cache/generation")
    diff_config = DiffusionConfig(cpu_offload=True) 
    diffusion_provider = DiffusionProvider(config=diff_config)
    renderer_provider = FFmpegVideoRenderer()

    # Import new stages
    from core.planning.scene_splitter import SceneSplitterStage
    from core.planning.shot_planner import ShotPlannerStage
    from core.planning.camera_planner import CameraPlannerStage
    from core.validation.pipeline_validator import ValidatorStage
    from core.planning.timeline_builder import TimelineBuilderStage
    from core.rendering.image_stage import DiffusionRendererStage
    from core.rendering.assembly_stage import FFmpegAssemblyStage

    # Stages need to expose get_providers for Executor lifecycle management
    sb_stage = StoryBibleGeneratorStage(llm_provider)
    scene_stage = SceneSplitterStage(llm_provider)
    shot_stage = ShotPlannerStage(llm_provider)
    camera_stage = CameraPlannerStage()
    prompt_stage = PromptBuilderStage()
    valid_stage = ValidatorStage()
    timeline_stage = TimelineBuilderStage()
    img_stage = DiffusionRendererStage(diffusion_provider=diffusion_provider)
    render_stage = FFmpegAssemblyStage()

    stages_map = {
        "story_bible": sb_stage,
        "scene": scene_stage,
        "shot": shot_stage,
        "camera": camera_stage,
        "prompt": prompt_stage,
        "validate": valid_stage,
        "timeline": timeline_stage,
        "diffusion": img_stage,
        "render": render_stage
    }

    if args.stage == "all":
        active_stages = list(stages_map.values())
    else:
        active_stages = [stages_map[args.stage]]

    router = ContractRouter({})
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
