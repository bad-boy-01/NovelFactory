# NovelFactory

NovelFactory is a pipeline-based generative compiler designed to automate the conversion of text (such as novels or scripts) into rendered videos ("visual novels"). It breaks down stories into chapters, scenes, and beats, uses AI to extract character consistency rules (a "Story Bible"), compiles stylistic prompts, generates images, evaluates them for quality, and finally stitches them together into an MP4 video sequence.

## How it Works

The system operates as a **replayable generative compiler** with a closed-loop feedback mechanism:

1. **Dataset Parsing (`core/pipeline/dataset_provider.py`)**: Ingests raw text and splits it into an Abstract Syntax Tree of Chapters, Scenes, and Beats.
2. **Story Bible Generation (`ai/reasoning/story_bible.py`)**: Uses an LLM to analyze the narrative and extract global visual continuity rules (e.g., character outfits, DNA, color palettes).
3. **Prompt Compilation (`ai/prompting/compiler.py`)**: Cross-references scene beats with the Story Bible and selected stylistic `PromptPack` to generate deterministic prompts.
4. **Media Generation (`ai/generation/image.py`)**: Pluggable architecture that calls out to Image Generation models (like Stable Diffusion) while recording exact generation lineage (Provenance Graph).
5. **Multi-Signal Evaluation (`ai/reasoning/evaluator.py`)**: Grades generated assets using multiple signals (CLIP, face embeddings) and decides whether to retry based on dynamic drift budgets.
6. **Video Rendering (`ai/generation/rendering.py`)**: Stitches the final approved images into video chapters.

The core architecture treats the pipeline state (`PipelineContext`) as a mutable/event-sourced graph where each stage fingerprint ensures perfect reproducibility and provenance tracking.

---

## Running in Kaggle

NovelFactory is designed to be run in cloud environments with GPU access like Kaggle. Since the core relies on abstract plugin interfaces, you will need to provide concrete implementations for the `ImageGeneratorProvider` and `LLMProvider` (e.g., using Hugging Face `diffusers` and `transformers`).

### Kaggle Cell Commands

Run the following commands in Kaggle notebook cells to get started:

**1. Clone the repository and navigate to the directory:**
```python
!git clone https://github.com/bad-boy-01/NovelFactory.git
%cd NovelFactory
```

**2. Install dependencies:**
The core framework requires `pydantic`. You will also need to install the dependencies for whatever AI models you plan to use as plugins.
```python
!pip install pydantic
# Install your preferred provider dependencies (e.g., diffusers, torch)
!pip install diffusers transformers torch accelerate
```

**3. Example Execution Framework:**
NovelFactory is an orchestration framework. To run it, you must implement the plugin interfaces and wire up the executor. Here is a boilerplate cell to get you started:

```python
from pathlib import Path
from core.pipeline.executor import SequentialExecutor
from core.pipeline.context import PipelineContext
from core.domain.project import ProjectManifest

# 1. Initialize your context
project_manifest = ProjectManifest(project_name="MyNovel", dataset_id="dataset_01")
context = PipelineContext(project_manifest=project_manifest)

# 2. Define your plugins (You must implement these interfaces in Kaggle)
# from plugins.interfaces import ImageGeneratorProvider, LLMProvider
# class MyHuggingFaceGenerator(ImageGeneratorProvider): 
#     ...
# class MyLLMProvider(LLMProvider): 
#     ...

# 3. Instantiate Stages
# from ai.reasoning.story_bible import StoryBibleGeneratorStage
# from ai.prompting.compiler import PromptCompilerStage
# from ai.generation.image import ImageGenerationStage

# stages = [
#     StoryBibleGeneratorStage(llm=MyLLMProvider()),
#     PromptCompilerStage(prompt_pack=my_pack),
#     ImageGenerationStage(provider=MyHuggingFaceGenerator(), output_dir=Path("./output"))
# ]

# 4. Run Pipeline
# executor = SequentialExecutor(stages=stages)
# final_context = executor.run(context)
# print("Pipeline execution complete!")
```
