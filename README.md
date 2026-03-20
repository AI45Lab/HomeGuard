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
<a href='https://arxiv.org/pdf/2603.14367'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://github.com/AI45Lab/HomeGuard'><img src='https://img.shields.io/badge/Project-Page-green'></a> <a href='https://huggingface.co/datasets/Ursulalala/HomeSafe'><img src='https://img.shields.io/badge/🤗-Dataset-blue'></a>
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

## ⚡ Quick Start

### 1. Install the Python environment

```bash
git clone https://github.com/AI45Lab/HomeGuard.git
cd HomeGuard

conda create -n homeguard python=3.11 -y
conda activate homeguard
pip install torch torchvision transformers peft openai pillow tqdm numpy pandas matplotlib opencv-python requests sentence-transformers scikit-learn
```

### 2. Download our released checkpoints

Download the released HomeGuard checkpoints from Hugging Face and place them under `checkpoints/`.

Placeholder links:
- `https://huggingface.co/Ursulalala/HomeGuard-4B`
- `https://huggingface.co/Ursulalala/HomeGuard-8B`

Recommended layout:

```text
checkpoints/
├── HomeGuard-4B
└── HomeGuard-8B
```

### 3. Download the HomeSafe test set

Download the HomeSafe test split from Hugging Face and place it under `data/homesafe/test`.

Placeholder link:
- `https://huggingface.co/datasets/Ursulalala/HomeSafe`

Recommended layout:

```bash
mkdir -p data/homesafe/test
# download and extract the test images into data/homesafe/test
```

## 🏗️ Data Construction

### 1. Download the edit model

Download `Qwen-Image-Edit-2511` and place it under `checkpoints/`.

Placeholder link:
- `https://huggingface.co/Qwen/Qwen-Image-Edit-2511`

Recommended layout:

```text
checkpoints/
└── Qwen-Image-Edit-2511
```

### 2. Configure API keys

Set the API endpoints and keys used by the pipeline nodes:

```bash
export PLAN_API_KEY=your_plan_api_key
export PLAN_API_URL=your_plan_api_url

export AUG_API_KEY=your_augmentation_api_key
export AUG_API_URL=your_augmentation_api_url

export REPLACE_API_KEY=your_replace_api_key
export REPLACE_API_URL=your_replace_api_url

export EDIT_API_KEY=your_edit_api_key
export EDIT_API_URL=your_edit_api_url

export VERIFY_API_KEY=your_verify_api_key
export VERIFY_API_URL=your_verify_api_url

export ANNOTATION_API_KEY=your_annotation_api_key
export ANNOTATION_API_URL=your_annotation_api_url
```

### 3. Prepare the HomeSafe seed data directory

Put the seed images and metadata under `data/homesafe/`.

Recommended layout:

```text
data/homesafe/
├── metadata/
├── edit_image/
└── test/
    ├── safe/
    └── unsafe/
```

### 4. Run the data pipeline

```bash
cd data/pipeline/nodes

python editing_planner.py   --planner_name Qwen/Qwen3-VL-235B-A22B-Thinking   --root_folder ../../homesafe   --max_workers 24

python obj_augmentation.py   --mode replace   --replace_model Qwen/Qwen3-VL-235B-A22B-Thinking   --root_folder ../../homesafe   --max_workers 24

python safe_scenario_generator.py   --model Qwen/Qwen3-VL-235B-A22B-Thinking   --root_folder ../../homesafe   --max-workers 24

python scene_editor.py   --scenario_type unsafe   --editor_model ../../../checkpoints/Qwen-Image-Edit-2511   --root_folder ../../homesafe   --max_workers 1

python fidelity_verifier.py   --scenario_type unsafe   --verifier_model Qwen/Qwen3-VL-235B-A22B-Thinking   --root_folder ../../homesafe   --max_workers 24

python hazard_verifier.py   --scenario_type unsafe   --detector_name Qwen/Qwen3-VL-235B-A22B-Thinking   --root_folder ../../homesafe   --max_workers 24

python object_state_annotator.py   --model Qwen/Qwen3-VL-235B-A22B-Thinking   --root_folder ../../homesafe   --max_workers 24

python cot_generator.py   --model Qwen/Qwen3-VL-235B-A22B-Thinking   --max_workers 24
```

## 🎓 Training

### 1. Download the training data

Download the HomeSafe training set from Hugging Face and place it under `data/homesafe/`.

Placeholder link:
- `https://huggingface.co/datasets/Ursulalala/HomeSafe`

### 2. Unzip the released image folders

After downloading the training data, extract the released image folders so that `data/homesafe/` contains:

```text
data/homesafe/
├── metadata/
├── edit_image/
│   ├── safe/
│   └── unsafe/
└── test/
```

### 3. Prepare third-party training frameworks

```bash
mkdir -p third_party
```

- Clone [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) to `third_party/LlamaFactory`
- Clone [Visual-RFT](https://github.com/om-ai-lab/Visual-RFT) to `third_party/Visual-RFT`

### 4. Prepare training checkpoints

Put the required backbone and reward checkpoints under `checkpoints/`.

Recommended layout:

```text
checkpoints/
├── all-MiniLM-L6-v2
├── Qwen3-VL-4B-Thinking
└── Qwen3-VL-8B-Thinking
```

### 5. Run SFT

```bash
export LLAMAFACTORY_ROOT=$PWD/third_party/LlamaFactory
export MODEL_PATH=$PWD/checkpoints/Qwen3-VL-4B-Thinking
bash training/sft/scripts/qwen3vl_4b_thinking_lora_step_sft.sh
```

### 6. Run GRPO / RFT

```bash
export VISUAL_ROOT_PATH=$PWD/third_party/Visual-RFT
export CKPT_PATH=/path/to/your/step-sft-checkpoint
bash training/grpo/scripts/train_rft_action_4b_step.sh
```

See [training/README.md](./training/README.md) for additional training assets and scripts.

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
