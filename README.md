# NovelFactory: Generative Compiler & Control System

NovelFactory is not just a media pipeline—it is a **replayable generative compiler** and a **closed-loop generative operating system**. It is designed to solve the fundamental problem of converting unstructured text (like novels or scripts) into temporally and visually coherent rendered videos ("visual novels") by treating generative failures as measurable system drift rather than isolated errors.

---

## 1. Core Philosophy

In traditional generative systems, the architecture assumes *same input → same output*. However, under real LLM and Diffusion variance, this breaks down. NovelFactory shifts the paradigm from deterministic pipelines to **stochastic execution engines with policy-driven correction loops**.

Instead of manually debugging images, NovelFactory is built on a **Generative Observability Stack** that formalizes:
* **Event-Sourced Execution:** Turning the pipeline into a replayable computation graph.
* **Multi-Signal Evaluation:** Using calibrated signals (CLIP + Face Embeddings + Attribute Classifiers) rather than generic thresholds.
* **Drift Budgets:** Allocating compute tolerance dynamically across Narrative, Visual, and Stylistic axes.

---

## 2. Architecture: The Generative Compiler

The system breaks down narratives similar to how a compiler processes source code:

1. **AST Construction (`DatasetProvider`)**: Parses `.txt` or `.docx` into an Abstract Syntax Tree (AST) of Chapters, Scenes, and Beats.
2. **Semantic Enrichment (`StoryBibleGeneratorStage`)**: Acts as global memory. It uses an LLM to extract character consistency rules (Visual DNA, Outfits, Color Palettes).
3. **IR Transformation (`PromptCompilerStage`)**: Cross-references narrative beats with the Story Bible and cinematic stylistic `PromptPack` policies to generate an Intermediate Representation (`DeclarativePrompt`).
4. **Lowering & Execution (`ImageGenerationStage`)**: Calls out to Image Generation models (e.g., Stable Diffusion) while recording an exact `ProvenanceGraph` (the debug symbol table for generated assets).
5. **Linking & Packaging (`RenderingStage`)**: Stitches the approved latent frames together into an MP4 sequence.

---

## 3. The Control Policy Layer (Self-Regulation)

A naive pipeline relies on a `score → threshold → retry` loop. NovelFactory implements a **Control Policy Layer** to actively regulate the system.

### A. Dynamic Drift Budgets
Drift budgets act as a resource allocation scheduler. For example:
* **Chapter 1 (World-building):** The scheduler enforces strict **Identity Lock**, allocating high compute retry budgets to ensure character faces perfectly match the Story Bible.
* **Chapter 3 (Action Sequence):** The scheduler relaxes Identity Lock in favor of **Motion Diversity**, allowing slight semantic drift to prioritize cinematic pacing.

### B. Multi-Signal Calibration
Raw CLIP scores are easily overpowered by stylistic tokens. The `EvaluationCalibrationLayer` calculates a weighted consistency vector:
* **Semantic Alignment (CLIP):** General scene vibe.
* **Identity Lock (Face Embeddings):** Identity stability (e.g., ArcFace).
* **Attribute Classifiers:** Structured truth (e.g., "Is the uniform red?").

### C. Failure Taxonomy & Interventions
When an asset fails evaluation, the system diagnoses the root cause rather than guessing:
* **Identity Drift (Prompt Dilution):** *Intervention:* Mutate the AST to drop 20% of stylistic weights and retry.
* **Identity Drift (Model Incapacity):** *Intervention:* Swap the diffusion provider or inject stronger ControlNet conditioning.
* **Prompt Impossibility:** *Intervention:* Hard stop circuit breaker to prevent infinite retry loops.

---

## 4. The Event-Sourced Generative Graph

To achieve true reproducibility across a stochastic system, `PipelineContext` is not randomly mutated. Every execution emits a `StageExecutionEvent` containing:
* Input hash state.
* The explicit mutation log.
* Artifact references and `ProvenanceGraph` linkage.

This allows developers to run **Cross-Run Comparisons (Regression Tests)**. If you upgrade from Diffusion Model V1 to V2, you can replay the exact deterministic event graph and visualize the shift in prompt phase space stability.

---

## 5. Usage & Execution Harness

NovelFactory provides a full CLI integration harness. It allows you to run a complete "story → video" pipeline in one command to verify orchestration correctness.

### Installation

```bash
git clone https://github.com/bad-boy-01/NovelFactory.git
cd NovelFactory
pip install -r requirements.txt
```

*(Note: The core framework requires `pydantic` and `python-docx`. Real execution requires you to install specific plugin dependencies like `diffusers` or `transformers`.)*

### Running the CLI (Mock vs Real Mode)

The CLI supports two execution modes via the `--mode` flag:

#### 1. Mock Mode (`--mode mock`)
This mode allows you to test the orchestration flow, failure taxonomy, and traceability **without requiring a GPU or API keys**.

```bash
python main.py --project-name MyProject --script-file my_script.txt --mode mock
```

**What happens in Mock Mode?**
* The system bypasses heavy AI computation and uses hardcoded mock plugins.
* It simulates Image Generation by creating dummy, empty `.png` files.
* It simulates Video Rendering by creating a fake `chapter_output.mp4`.
* It simulates LLM reasoning by injecting hardcoded character data and occasional "hallucinations".
* It deliberately injects occasional evaluation failures (e.g., simulated "Identity Drift" or CUDA OOM errors) to test the pipeline's retry and telemetry logic.
* You get a complete execution trace in your console verifying that the architecture flows correctly from text ingestion to video stitching.

#### 2. Real Mode (`--mode real`)
This mode is meant for **actual video production**.

**Important:** Running with `--mode real` out-of-the-box will fail because NovelFactory is an architectural framework. To use real mode, you must first implement the concrete plugin classes in `plugins/interfaces.py` (e.g., writing a `HuggingFaceImageGenerator` that actually connects to Stable Diffusion via the `diffusers` library, or an `OpenAILLMProvider` that calls GPT-4). Once implemented, this mode will execute the heavy ML workloads on your GPU and generate real visual media.

### Extending with Plugins

To move from Mock Mode to Production, implement the core interfaces defined in `plugins/interfaces.py`:
* `LLMProvider`
* `ImageGeneratorProvider`
* `EvaluatorPlugin`
* `VideoRendererProvider`

By fulfilling these contracts, the Generative Compiler will automatically orchestrate, evaluate, and self-regulate your models.
