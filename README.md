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
