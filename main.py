import argparse
import logging
import sys
import shutil
from pathlib import Path
from docx import Document

from core.pipeline.project_builder import ProjectBuilder
from core.pipeline.context import PipelineContext
from core.pipeline.executor import SequentialExecutor

from ai.reasoning.story_bible import StoryBibleGeneratorStage
from ai.prompting.compiler import PromptCompilerStage
from ai.generation.image import ImageGenerationStage
from ai.reasoning.evaluator import EvaluationStage
from ai.generation.rendering import RenderingStage

from core.profiles.prompt_pack import PromptPack
from plugins.mock_providers import MockLLMProvider, MockImageGenerator, MockEvaluator, MockVideoRenderer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("NovelFactoryCLI")

def extract_text(file_path: Path) -> str:
    if file_path.suffix.lower() == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    elif file_path.suffix.lower() == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def main():
    parser = argparse.ArgumentParser(description="NovelFactory End-to-End Pipeline Harness")
    parser.add_argument("--project-name", required=True, help="Name of the project")
    parser.add_argument("--script-file", required=True, help="Path to the input script (.txt or .docx)")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock", help="Execution mode (mock or real)")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info({"event": "cli_start", "project": args.project_name, "mode": args.mode})

    # 1. Setup Workspace Paths
    workspace_dir = Path("./workspace")
    datasets_dir = workspace_dir / "datasets"
    projects_dir = workspace_dir / "projects"
    runs_dir = workspace_dir / "runs"
    
    for d in [datasets_dir, projects_dir, runs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    script_path = Path(args.script_file)
    if not script_path.exists():
        logger.error(f"Script file not found: {script_path}")
        sys.exit(1)

    # 2. Extract & Normalize Dataset
    try:
        raw_text = extract_text(script_path)
    except Exception as e:
        logger.error({"event": "text_extraction_failed", "error": str(e)})
        sys.exit(1)

    dataset_path = datasets_dir / args.project_name
    dataset_path.mkdir(exist_ok=True)
    with open(dataset_path / "novel.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)
    
    logger.info({"event": "dataset_created", "path": str(dataset_path)})

    # 3. Build Project Manifest
    builder = ProjectBuilder(projects_dir=projects_dir, datasets_dir=datasets_dir)
    manifest = builder.build_project(project_name=args.project_name, dataset_name=args.project_name)

    # 4. Create Pipeline Context
    context = PipelineContext(project_manifest=manifest)

    # 5. Instantiate Providers
    if args.mode == "real":
        logger.error("Real mode is not yet implemented. Please run with --mode mock")
        sys.exit(1)
        
    llm_provider = MockLLMProvider()
    image_provider = MockImageGenerator()
    evaluator_plugin = MockEvaluator()
    video_provider = MockVideoRenderer()
    
    default_prompt_pack = PromptPack(
        name="Cinematic Anime",
        positive_prefix="anime style, highly detailed, cinematic lighting",
        negative_prompt="bad anatomy, lowres, missing fingers"
    )

    project_output_dir = projects_dir / args.project_name / "exports"

    # 6. Build Stages
    stages = [
        StoryBibleGeneratorStage(llm=llm_provider),
        PromptCompilerStage(prompt_pack=default_prompt_pack),
        ImageGenerationStage(provider=image_provider, output_dir=projects_dir / args.project_name / "assets"),
        EvaluationStage(evaluators=[evaluator_plugin], threshold=0.8),
        RenderingStage(renderer=video_provider, output_dir=project_output_dir)
    ]

    # 7. Execute Pipeline
    # To run the pipeline over chapters, we need to manually invoke the DatasetProvider MVP logic 
    # since the executor doesn't natively iterate over chapters yet.
    from core.pipeline.dataset_provider import LocalDatasetProvider
    provider = LocalDatasetProvider(dataset_path)
    
    # Simple runner loop
    for chapter in provider.iter_chapters():
        context.current_chapter = chapter
        for scene in chapter.scenes:
            context.current_scene = scene
            
            executor = SequentialExecutor(stages=stages)
            try:
                context = executor.run(context)
            except Exception as e:
                logger.error({"event": "pipeline_failure", "error": str(e)})

    # 8. Report Final Status
    logger.info("========== PIPELINE EXECUTION SUMMARY ==========")
    logger.info(f"Project Name: {context.project_manifest.project_name}")
    logger.info(f"Generated Assets: {len(context.assets)}")
    
    eval_failures = [a for a in context.assets.values() if getattr(a, 'evaluation', None) and getattr(a.evaluation, 'retry_needed', False)]
    logger.info(f"Assets Needing Retry (Failed Eval): {len(eval_failures)}")
    
    for a in context.assets.values():
        logger.info(f"- Asset ID: {a.id} | Type: {a.asset_type.value} | Path: {a.file_path}")

if __name__ == "__main__":
    main()
