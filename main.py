import argparse
import logging
import json
import sys
from pathlib import Path
import shutil

# Force unbuffered output so every print/log line appears immediately in Kaggle
sys.stdout.reconfigure(line_buffering=True)

from core.pipeline.context import PipelineContext
from core.domain.story.project import ProjectManifest, ProjectMetadata
from plugins.local_llm import LocalLLMProvider
from core.planning.story_bible_stage import StoryBibleGeneratorStage
from plugins.local_diffusion import LocalDiffusionProvider
from plugins.interfaces import DiffusionConfig
from core.pipeline.cache import CacheProvider
from core.optimization.prompt_builder import PromptBuilderStage
from plugins.ffmpeg_renderer import FFmpegVideoRenderer
from core.rendering.assembly_stage import FFmpegAssemblyStage
from core.rendering.executor import SequentialExecutor
from core.contracts.router import ContractRouter

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("NovelFactory")

def save_workspace(name: str, content: str):
    Path("workspace").mkdir(exist_ok=True)
    with open(f"workspace/{name}", "w", encoding="utf-8") as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description="NovelFactory Milestone 1 Validator")
    parser.add_argument("--novel", type=str, default="sample.txt", help="Path to novel text")
    parser.add_argument("--stage", type=str, default="all", help="Stage to run: story_bible, prompt, diffusion, render, all, validate")
    parser.add_argument("--mode", type=str, default="all", choices=["plan", "render", "all"], help="Execution mode: plan, render, all")
    parser.add_argument("--render-shot", type=str, help="Specific shot to render, e.g., '31'")
    parser.add_argument("--render-scene", type=str, help="Specific scene to render, e.g., '4'")
    parser.add_argument("--render-shots", type=str, help="Range of shots to render, e.g., '20-28'")
    args = parser.parse_args()

    if args.stage == "validate":
        logger.info("[VALIDATE] Running architectural sanity check...")
        manifest = ProjectManifest(metadata=ProjectMetadata(project_name="Sanity", dataset_id="test"), source_text="test")
        context = PipelineContext(project_manifest=manifest)
        llm = LocalLLMProvider()
        cache = CacheProvider(".cache")
        diff = LocalDiffusionProvider(DiffusionConfig(cpu_offload=True))
        rend = FFmpegVideoRenderer()
        from core.rendering.image_stage import DiffusionRendererStage
        stages = [StoryBibleGeneratorStage(llm), PromptBuilderStage(), DiffusionRendererStage(diffusion_provider=diff), FFmpegAssemblyStage()]
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

    manifest = ProjectManifest(metadata=ProjectMetadata(project_name="Milestone1", dataset_id="local_run"), source_text=novel_text)
    context = PipelineContext(project_manifest=manifest)

    llm_provider = LocalLLMProvider() 
    cache_provider = CacheProvider(cache_dir=".cache/generation")
    diff_config = DiffusionConfig(cpu_offload=True) 
    diffusion_provider = LocalDiffusionProvider(config=diff_config)
    renderer_provider = FFmpegVideoRenderer()

    from core.domain.workspace import WorkspaceManager
    from core.domain.assets.registry import AssetRegistry
    from core.pipeline.compiler_context import CompilerContext
    
    workspace = WorkspaceManager(base_dir="workspace")
    registry = AssetRegistry()
    registry.load(workspace)
    
    compiler_context = CompilerContext(
        pipeline_context=context,
        workspace=workspace,
        registry=registry
    )

    # Import new stages
    from core.planning.chunker import ChunkerStage
    from core.planning.scene_splitter import SceneSplitterStage
    from core.planning.shot_planner import ShotPlannerStage
    from core.planning.camera_planner import CameraPlannerStage
    from core.validation.pipeline_validator import ValidatorStage
    from core.planning.timeline_builder import TimelineBuilderStage
    from core.rendering.image_stage import DiffusionRendererStage
    from core.rendering.assembly_stage import FFmpegAssemblyStage

    # Stages need to expose get_providers for Executor lifecycle management
    sb_stage = StoryBibleGeneratorStage(llm_provider)
    chunk_stage = ChunkerStage(chunk_size=2000, overlap=200)
    scene_stage = SceneSplitterStage(llm_provider)
    shot_stage = ShotPlannerStage(llm_provider)
    camera_stage = CameraPlannerStage()
    prompt_stage = PromptBuilderStage()
    valid_stage = ValidatorStage()
    timeline_stage = TimelineBuilderStage()
    render_options = {
        "shot": args.render_shot,
        "scene": args.render_scene,
        "shots": args.render_shots
    }
    img_stage = DiffusionRendererStage(diffusion_provider=diffusion_provider, render_options=render_options)
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

    planning_stages = [sb_stage, chunk_stage, scene_stage, shot_stage, camera_stage, prompt_stage, valid_stage, timeline_stage]
    rendering_stages = [img_stage, render_stage]
    
    if args.mode == "plan":
        active_stages = planning_stages
    elif args.mode == "render":
        active_stages = rendering_stages
    elif args.mode == "all":
        if args.stage != "all":
            active_stages = [stages_map[args.stage]]
        else:
            active_stages = planning_stages + rendering_stages

    router = ContractRouter({})
    from core.pipeline.resource_manager import ResourceSession

    try:
        final_context = compiler_context
        
        if args.mode in ["plan", "all"] and args.stage == "all":
            executor_plan = SequentialExecutor(stages=planning_stages, contract_router=router, max_retries=2)
            with ResourceSession(resources=[llm_provider]):
                final_context = executor_plan.run(final_context)
                
        if args.mode in ["render", "all"] and args.stage == "all":
            executor_render = SequentialExecutor(stages=rendering_stages, contract_router=router, max_retries=2)
            # Diffusion residency deferred to Phase 4.3
            final_context = executor_render.run(final_context)
            
        if args.stage != "all":
            executor_single = SequentialExecutor(stages=active_stages, contract_router=router, max_retries=2)
            with ResourceSession(resources=[llm_provider]): # Best effort for single stage
                final_context = executor_single.run(final_context)

        
        # Save intermediate artifacts
        for node in final_context.execution_nodes:
            art = node.artifact
            name = type(art).__name__
            if name in ["StoryBible", "SceneManifest", "ShotManifest", "PromptManifest", "Timeline"]:
                workspace.save_json(f"{name}.json", json.loads(art.model_dump_json()))
                
        # Save updated registry
        registry.save(workspace)
                
        if args.mode in ["plan", "all"]:
            from core.reporting.reporter import CompilerReporter
            reporter = CompilerReporter()
            report_data = reporter.generate_compile_report(final_context.pipeline)
            workspace.save_json("compile_report.json", report_data)
            
            director_report = reporter.generate_directors_report(final_context.pipeline)
            with open(workspace.get_output_path("director_report.txt"), "w", encoding="utf-8") as f:
                f.write(director_report)
            print("\n" + director_report)
        
        # Copy workspace to debug
        shutil.copytree("workspace", "debug/workspace_run", dirs_exist_ok=True)

        logger.info("\n[SUCCESS] Pipeline execution complete.")
    except Exception as e:
        logger.error(f"\n[ERROR] Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()
