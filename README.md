<div align="center">
  <h1 style="display: inline-block; margin: 0;"> HomeGuard: VLM-based Embodied Safeguard for Identifying Contextual Risk in Household Task</h1>
</div>

<h4 align="center"> 

Xiaoya Lu<sup>1,2*</sup>,
Yijin Zhou<sup>1,2*</sup>,
Zeren Chen<sup>3,1</sup>,
Ruocheng Wang<sup>2</sup>,
Bingrui Sima<sup>4</sup>, <br>
Enshen Zhou<sup>3</sup>,
Lu Sheng<sup>3</sup>,
Dongrui Liu<sup>1✉</sup>,
Jing Shao<sup>1✉</sup>

<sup>1</sup>Shanghai AI Laboratory, <sup>2</sup>Shanghai Jiao Tong University, <br>
<sup>3</sup>Beihang University, <sup>4</sup>Huazhong University of Science and Technology

*Equal Contribution, ✉Corresponding authors

</h4>

<div align="center">
<a href='https://arxiv.org/pdf/2603.14367'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://github.com/AI45Lab/HomeGuard'><img src='https://img.shields.io/badge/Project-Page-green'></a> <a href='https://huggingface.co/datasets/Ursulalala/'><img src='https://img.shields.io/badge/🤗-Dataset-blue'></a>
</a>
</div>


## 📰 News
* **`2026.03.16`** 🤗🤗 We release our latest work [HomeGuard](https://arxiv.org/pdf/2603.14367), the **first specialized embodied safeguard model** for identifying contextual risk in household task.
* 🚀 Code release in progress! We are currently organizing the repository and will open-source it soon.

## 👀 Overview

- 🤖 **Addressing Implicit Contextual Risks:** While explicit malicious instructions are easier to detect, embodied agents often fail to identify **implicit contextual risks**—where benign instructions (e.g., "heat food") become hazardous due to environmental states (e.g., metal in a microwave).
- 🛡️ **Architecture-Agnostic Safeguard:** We propose **HomeGuard**, a plug-and-play safeguard that avoids complex rule-based systems. It uses **Context-Guided Chain-of-Thought (CG-CoT)** to decompose safety into *active perception* (prioritizing interaction targets) and *semantic risk judgment*.
- 🎯 **Visual Anchors for Grounding:** By equipping VLMs with visual anchors (bounding boxes), HomeGuard directs attention to risk-critical regions, effectively mitigating hallucinations and "unfocused perception" in cluttered, object-dense scenes.

<p align='center'>
<img src='./assets/teaser.png' alt='Teaser' width='850px'>
  <br>
  <em><b>Figure 1: Identifying implicit contextual risks via Context-Guided Chain-of-Thought.</b></em>
</p>

<p align='center'>
<img src='./assets/safe_trajectory.png' alt='Trajectory' width='70%'>
  <br>
  <em><b>Figure 2: An application case of HomeGuard facilitating safe trajectory generation. </b></em>
</p>

## 📊 Performance

- 🚀 **State-of-the-Art Risk Identification:** HomeGuard-8B achieves a **90.98% RIR** and **74.90% RMR** on HomeSafe-Bench, significantly outperforming leading open-source models (Qwen3-VL-235B) and even matching or surpassing proprietary models like Gemini-3-Pro in complex embodied scenarios.
- 📉 **Significant Reduction in Oversafety:** By prioritizing hazard regions through active perception, HomeGuard reduces the oversafety rate by up to **19.48%**, ensuring the agent remains functional without being overly cautious or "paranoid" due to perceptual noise.
- 🌍 **Superior Generalization:** Beyond our benchmark, HomeGuard demonstrates robust performance on four public risk identification benchmarks (EARBench, MSSBench, etc.), delivering results comparable to GPT-4o-mini and improving risk prediction accuracy by over 40% compared to base models.
- 🛠️ **Practical Utility for Safe Planning:** Integrating HomeGuard into VLM planners yields a **16.11% improvement** on the IS-Bench safe success rate. Beyond semantic risk grounding, the generated bounding boxes serve as **actionable spatial waypoints**, enabling low-level safe trajectory generation.

<p align='center'>
<img src='./assets/experiment1.png' alt='Table 1' width='850px'>
</p>

<p align='center'>
<img src='./assets/experiment2.png' alt='Table 2' width='850px'>
</p>

<p align='center'>
<img src='./assets/efficiency.png' alt='Table 3' width='850px'>
</p>

## ⚙️ Installation

HomeGuard depends on different upstream frameworks for different stages. We only keep HomeGuard-specific code in this repository.

### 1. Clone this repository

```bash
git clone https://github.com/AI45Lab/HomeGuard.git
cd HomeGuard
```

### 2. Create a base environment for evaluation and application

```bash
conda create -n homeguard python=3.11 -y
conda activate homeguard
pip install torch torchvision transformers peft openai pillow tqdm numpy pandas matplotlib opencv-python requests sentence-transformers scikit-learn
```

### 3. Prepare optional third-party dependencies

```bash
mkdir -p third_party
```

- For SFT, clone [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) into `third_party/LlamaFactory` or any path you prefer.
- For GRPO, clone [Visual-RFT](https://github.com/om-ai-lab/Visual-RFT) into `third_party/Visual-RFT` or any path you prefer.
- For trajectory generation, place RoboBrain2.5 under `third_party/Robobrain2.5` or pass `--third-party-root` explicitly.

### 4. Prepare checkpoints

Create a local checkpoint directory and download the required models before running training or the data construction pipeline.

```bash
mkdir -p checkpoints
```

Recommended layout:

- `checkpoints/all-MiniLM-L6-v2` for the GRPO reward model
- `checkpoints/Qwen3-VL-4B-Thinking` for 4B SFT / GRPO training
- `checkpoints/Qwen3-VL-8B-Thinking` for 8B SFT / GRPO training
- `checkpoints/Qwen-Image-Edit-2511` for data construction and image editing

Required checkpoints by use case:

- Training: `all-MiniLM-L6-v2`, `Qwen3-VL-4B-Thinking`, `Qwen3-VL-8B-Thinking`
- Data construction: `Qwen-Image-Edit-2511`

### 5. Prepare non-redistributed assets

This release does not bundle large image assets. Put them into the placeholder directories documented in [data/README.md](./data/README.md).

- HomeGuard assets go under `data/homeguard/`, with images in `base_image/`, `edit_image/`, `annotate_image/`, and `test/`, and metadata in `data/homeguard/metadata/`
- Public benchmark metadata is bundled under `data/public_benches/`, while benchmark images should be downloaded separately from Hugging Face
- Optional local checkpoints can go under `checkpoints/`

## 🗂️ Data

### Download HomeSafe dataset
homeguard/
├── metadata/
├── base_image/
├── annotate_image/
│   ├── safe/            
│   └── unsafe/
├── edit_image/
│   ├── safe/            
│   └── unsafe/
└── test/

### Data pipeline usage

Run the node scripts in the following order from the repository root:

```bash
cd data/pipeline/nodes

python editing_planner.py \
  --planner_name Qwen/Qwen3-VL-235B-A22B-Thinking \
  --root_folder ../../homeguard \
  --max_workers 24

python obj_augmentation.py \
  --mode replace \
  --replace_model Qwen/Qwen3-VL-235B-A22B-Thinking \
  --root_folder ../../homeguard \
  --max_workers 24

python safe_scenario_generator.py \
  --model Qwen/Qwen3-VL-235B-A22B-Thinking \
  --root_folder ../../homeguard \
  --max-workers 24

python scene_editor.py \
  --scenario_type unsafe \
  --editor_model ../../../checkpoints/Qwen-Image-Edit-2511 \
  --root_folder ../../homeguard \
  --max_workers 1

python fidelity_verifier.py \
  --scenario_type unsafe \
  --verifier_model Qwen/Qwen3-VL-235B-A22B-Thinking \
  --root_folder ../../homeguard \
  --max_workers 24

python hazard_verifier.py \
  --scenario_type unsafe \
  --detector_name Qwen/Qwen3-VL-235B-A22B-Thinking \
  --root_folder ../../homeguard \
  --max_workers 24

python object_state_annotator.py \
  --model Qwen/Qwen3-VL-235B-A22B-Thinking \
  --root_folder ../../homeguard \
  --max_workers 24

python cot_generator.py \
  --model Qwen/Qwen3-VL-235B-A22B-Thinking \
  --max_workers 24
```

If you also want to generate the safe counterpart branch, run:

```bash
cd data/pipeline/nodes

python scene_editor.py \
  --scenario_type safe \
  --editor_model ../../../checkpoints/Qwen-Image-Edit-2511 \
  --root_folder ../../homeguard \
  --max_workers 1

python fidelity_verifier.py \
  --scenario_type safe \
  --verifier_model Qwen/Qwen3-VL-235B-A22B-Thinking \
  --root_folder ../../homeguard \
  --max_workers 24

python hazard_verifier.py \
  --scenario_type safe \
  --detector_name Qwen/Qwen3-VL-235B-A22B-Thinking \
  --root_folder ../../homeguard \
  --max_workers 24
```

## 🎓 Training

### Stage 1: SFT with LlamaFactory
The SFT assets live in `training/sft/`. We provide HomeGuard dataset files plus launch scripts that copy the required JSON files into your [LlamaFactory](https://github.com/hiyouga/LlamaFactory) checkout before training.

Example:

```bash
export LLAMAFACTORY_ROOT=$PWD/third_party/LlamaFactory
export MODEL_PATH=/path/to/Qwen3-VL-4B-Thinking
bash training/sft/scripts/qwen3vl_4b_thinking_lora_step_sft.sh
```

### Stage 2: GRPO / RFT with Visual-RFT

The GRPO assets live in `training/grpo/`. We keep the HomeGuard-specific reward code and training entrypoint here, while the general trainer is expected from [Visual-RFT](https://github.com/Liuziyu77/Visual-RFT).

Example:

```bash
export VISUAL_ROOT_PATH=$PWD/third_party/Visual-RFT
export CKPT_PATH=/path/to/your/step-sft-checkpoint
bash training/grpo/scripts/train_rft_action_4b_step.sh
```

See [training/README.md](./training/README.md) for the split between SFT and GRPO assets.

## 📏 Evaluation

The evaluation module supports both our HomeSafe-Bench and four public benchmarks.

### HomeSafe-Bench

```bash
export TARGET_MODEL=/path/to/model-or-api-name
export TARGET_API_URL=http://your-target-endpoint/v1
export TARGET_API_KEY=your-target-key
export EVALUATION_API_URL=http://your-judge-endpoint/v1
export EVALUATION_API_KEY=your-judge-key
python -m evaluation.evaluation --target_model "$TARGET_MODEL" --version v1
```

### Public benchmarks

Download benchmark images before running the public benchmark evaluators:

- EARBench images: https://huggingface.co/datasets/ZihaoZhu/EARDataset
- MSSBench images: https://huggingface.co/datasets/kzhou35/mssbench/
- PaSBench images: https://huggingface.co/datasets/Youliang/PaSBench/

Expected locations:
- `data/public_benches/earbench/images/`
- `data/public_benches/mssbench/embodied/`
- `data/public_benches/pasbench/combine_images/`

For SafeAgentBench, we only evaluate risk identification. After initializing each SafeAgentBench scene in the simulator once, we save the rendered screenshots and evaluate directly on those images `data/public_benches/sabench/images/`, so the simulator does not need to be redeployed for every run.

```bash
python -m evaluation.eval_earbench --target_model "$TARGET_MODEL" --version v1
```

We also provide wrapper scripts in `evaluation/scripts/` that read the same environment variables.

## 🤖 Application

The application module shows how to connect HomeGuard with downstream planners.

### 1. Convert HomeGuard outputs into safe plans

```bash
export PLANNER_API_URL=http://your-planner-endpoint/v1
export PLANNER_API_KEY=your-planner-key
python application/plan_traj.py \
  --image /path/to/scene.png \
  --task "Pour tea into the teacup next to the laptop" \
  --bbox-json /path/to/homeguard_prediction.json \
  --safety-tips "Avoid spilling liquid onto the laptop"
```

### 2. Render or replay low-level trajectories with RoboBrain

```bash
export ROBOBRAIN_MODEL_PATH=/path/to/RoboBrain2.5-checkpoint
python application/robo_traj.py \
  --image /path/to/scene.png \
  --prompt "Move to the handle of the microwave" \
  --plot
```

## 📝 Notes

- This repository reorganizes the original internal research code into four open-source-friendly modules: data, training, evaluation, and application.
- The original source repository is left untouched; all files here were copied or adapted inside the public `HomeGuard` repo only.
- Large image assets are intentionally omitted to keep the repository lightweight.
