<h1 align="center">XModBench</h1>

<p align="center">
  <b>Benchmarking Cross-Modal Capabilities and Consistency in Omni-Language Models</b>
</p>

<p align="center">
  <a href="https://iclr.cc/Conferences/2026"><img src="https://img.shields.io/badge/ICLR-2026-8e44ad.svg" alt="ICLR 2026"></a>
  <a href="https://arxiv.org/abs/2510.15148"><img src="https://img.shields.io/badge/Arxiv-2510.15148-b31b1b.svg" alt="Paper"></a>
  <a href="https://xingruiwang.github.io/projects/XModBench/"><img src="https://img.shields.io/badge/Website-Page-0a7aca?logo=globe&logoColor=white" alt="Website"></a>
  <a href="https://huggingface.co/datasets/RyanWW/XModBench"><img src="https://img.shields.io/badge/HuggingFace-Dataset-FFD21E?logo=huggingface" alt="Dataset"></a>
  <a href="https://github.com/XingruiWang/XModBench"><img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white" alt="GitHub Repo"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <img src="https://xingruiwang.github.io/projects/XModBench/static/images/teaser.png" width="92%" alt="XModBench teaser">
</p>

<p align="center">
  <i>🎉 Accepted at <b>ICLR 2026</b></i>
</p>


XModBench is a comprehensive benchmark designed to evaluate the cross-modal capabilities and consistency of omni-language models. It systematically assesses model performance across multiple modalities (text, vision, audio) and various cognitive tasks, revealing critical gaps in current state-of-the-art models.

### Key Features

- **🎯 Multi-Modal Evaluation**: Comprehensive testing across text, vision, and audio modalities
- **🧩 5 Task Dimensions**: Perception, Spatial, Temporal, Linguistic, and External Knowledge tasks
- **📊 13 SOTA Models Evaluated**: Including Gemini 2.5 Pro, Qwen2.5-Omni, EchoInk-R1, and more
- **🔄 Consistency Analysis**: Measures performance stability across different modal configurations
- **👥 Human Performance Baseline**: Establishes human-level benchmarks for comparison


## 📂 Dataset

The dataset is available on Hugging Face: [RyanWW/XModBench](https://huggingface.co/datasets/RyanWW/XModBench)

### Task Groups and Subtasks

<!-- | Group | Subtasks | Samples |
|---|---|---:|
| Perception | finegrained, general_activities, instruments, instruments_comp, natures | 27,000 |
| Spatial | 3D_movements, arrangements, panaroma | 7,791 |
| Speech | recognition, translation | 8,244 |
| Temporal | calculation, count, order | 9,000 |
| External Knowledge | emotion_classification, movie_matching, music_genre_classification, singer_identification | 12,300 |
| **Total** | **17 subtasks** | **64,335** | -->

| Group | Subtask | Samples |
|---|---|---:|
| Perception | finegrained | 6,000 |
| Perception | general_activities | 6,000 |
| Perception | instruments | 6,000 |
| Perception | instruments_comp | 3,000 |
| Perception | natures | 6,000 |
| **Perception total** | | **27,000** |
| Spatial | 3D_movements | 2,646 |
| Spatial | arrangements | 2,790 |
| Spatial | panaroma | 2,355 |
| **Spatial total** | | **7,791** |
| Speech | recognition | 4,032 |
| Speech | translation | 4,212 |
| **Speech total** | | **8,244** |
| Temporal | calculation | 3,000 |
| Temporal | count | 3,000 |
| Temporal | order | 3,000 |
| **Temporal total** | | **9,000** |
| External | emotion_classification | 4,200 |
| External | movie_matching | 1,200 |
| External | music_genre_classification | 6,000 |
| External | singer_identification | 900 |
| **External total** | | **12,300** |
| **Grand total** | | **64,335** |

### Modality Combinations

The benchmark covers all combinations of three modalities — **Audio**, **Vision** (image or video), and **Text** — as condition and answer options:

| Condition → Options | Samples |
|---|---:|
| Audio → Vision | 10,720 |
| Audio → Text | 10,720 |
| Vision → Audio | 10,725 |
| Vision → Text | 10,725 |
| Text → Audio | 10,725 |
| Text → Vision | 10,720 |

### Repository Structure

```
XModBench/
├── benchmark/
│   ├── Data/                        # Raw media files (audio, image, video)
│   │   ├── vggss_audio_bench/       #   VGGSound audio clips
│   │   ├── landscape_audiobench/    #   Landscape images
│   │   ├── emotions/                #   Emotion classification media
│   │   └── ...
│   ├── tasks/                       # Source QA JSON files, organised by subtask
│   │   ├── 01_perception/
│   │   │   ├── finegrained/         #   6 modality-combo JSON files, 1000 instances each
│   │   │   ├── general_activities/
│   │   │   ├── instruments/
│   │   │   ├── instruments_comp/
│   │   │   └── natures/
│   │   ├── 02_spatial/
│   │   ├── 03_speech/
│   │   ├── 04_temporal/
│   │   └── 05_Exteral/
│   └── results/                     # Model evaluation results
├── models/                          # Model inference scripts
│   ├── Qwen2.5-Omni/
│   ├── Genimi/
│   ├── InternVL/
│   └── ...
└── scripts/                         # Helper scripts
```

## 🚀 Quick Start

### Basic Usage (legacy API-based evaluation)

```bash


#!/bin/bash
#SBATCH --job-name=VLM_eval        
#SBATCH --output=log/job_%j.out
#SBATCH --error=log/job_%j.log                        
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

module load conda
# conda activate vlm
conda activate omni

export audioBench='/home/xwang378/scratch/2025/AudioBench'

# python $audioBench/scripts/run.py \
#     --model gemini \
#     --task_name perception/vggss_audio_vision \
#     --sample 1000


# python $audioBench/scripts/run.py \
#     --model gemini \
#     --task_name perception/vggss_vision_audio \
#     --sample 1000

# python $audioBench/scripts/run.py \
#     --model gemini \
#     --task_name perception/vggss_vision_text \
#     --sample 1000

# python $audioBench/scripts/run.py \
#     --model gemini \
#     --task_name perception/vggss_audio_text \
#     --sample 1000

# Qwen2.5-Omni

# python $audioBench/scripts/run.py \
#         --model qwen2.5_omni \
#         --task_name perception/vggss_audio_text \
#         --sample 1000

python $audioBench/scripts/run.py \
        --model qwen2.5_omni \
        --task_name perception/vggss_vision_text \
        --sample 1000


```



## 🔁 Reproduce with lmms-eval

We provide a fully reproducible evaluation path through [**lmms-eval**](https://github.com/XingruiWang/lmms-eval) (fork with XModBench tasks pre-integrated). The dataset is auto-downloaded from the [HF Hub](https://huggingface.co/datasets/RyanWW/XModBench) — no manual data prep.

> **Why dedicated model wrappers?** Each XModBench item places media in **both** the question stem **and every answer option** (up to 5 media per item). lmms-eval's *simple* model interface only attaches one media object per request, so omni models would silently see just the first media. We therefore add chat-style `*_interleave` wrappers that feed the full interleaved prompt to the model. **No upstream model file is modified.**

### 1. Install

```bash
git clone https://github.com/XingruiWang/lmms-eval.git
cd lmms-eval
pip install -e ".[all]"
```

### 2. Quick test (single config, 8 samples)

```bash
python -m lmms_eval \
    --model qwen2_5_omni_interleave \
    --model_args pretrained=Qwen/Qwen2.5-Omni-7B,device_map=auto,attn_implementation=flash_attention_2 \
    --tasks xmod_bench_lite_a2t \
    --batch_size 1 --limit 8 --log_samples \
    --output_path logs/debug
```

### 3. XModBench-Lite — 6,000 samples (5 families × 6 configs × 200)

`submit_lite.sh` launches all 6 modality configs with a resource-aware GPU
profile (no-video configs on 1 GPU, video configs on 4) so the full sweep
fits one QoS allocation:

```bash
# Qwen2.5-Omni-7B
./submit_lite.sh qwen2_5_omni_interleave Qwen/Qwen2.5-Omni-7B qwenomni3

# Qwen3-Omni-30B-A3B (MoE; all configs need 4 GPU)
LIGHT_GRES=gpu:a5000:4 HEAVY_GRES=gpu:a5000:4 \
  ./submit_lite.sh qwen3_omni_interleave Qwen/Qwen3-Omni-30B-A3B-Instruct qwenomni3 \
  device_map=auto,attn_implementation=flash_attention_2

# Level-2 metrics (by-config / by-family / disparity / imbalance)
python lmms_eval/tasks/xmod_bench/summarize.py \
    --logs logs/xmod_bench_lite/results_qwen2_5_omni_interleave/
```

### 4. Full benchmark — 61,320 samples (10 modality combinations)

```bash
TASKS=(xmod_bench_audio_text xmod_bench_text_audio \
       xmod_bench_audio_image xmod_bench_image_audio \
       xmod_bench_image_text xmod_bench_text_image \
       xmod_bench_audio_video xmod_bench_text_video \
       xmod_bench_video_audio xmod_bench_video_text)

python -m lmms_eval \
    --model qwen2_5_omni_interleave \
    --model_args pretrained=Qwen/Qwen2.5-Omni-7B,device_map=auto,attn_implementation=flash_attention_2 \
    --tasks "${TASKS[$SLURM_ARRAY_TASK_ID]}" \
    --batch_size 1 --log_samples \
    --output_path logs/xmod_bench_full/results
```

### Reproduction results (Qwen series)

By-config accuracy on **XModBench-Lite** via lmms-eval, vs. the paper's
full-set numbers (Table 2). Δ = Lite − paper.

| Config | Qwen2.5-Omni (Lite) | paper (full) | Δ | Qwen3-Omni (Lite) |
|--------|--------------------:|-------------:|----:|------------------:|
| Audio → Text   | 63.1 | 62.0 | **+1.1** | 71.6 |
| Audio → Vision | 49.8 | 48.0 | **+1.8** | 52.0 |
| Text → Audio   | 59.2 | 55.4 | **+3.8** | 62.5 |
| Text → Vision  | 62.5 | 59.6 | **+2.9** | 67.0 |
| Vision → Audio | 50.3 | 50.5 | **−0.2** | 55.6 |
| Vision → Text  | 76.4 | 76.3 | **+0.1** | 83.1 |

- **Qwen2.5-Omni reproduces the paper within |Δ| < 5 on all 6 configurations**
  on the lightweight 6k Lite split — confirming the lmms-eval port is faithful.
- **Qwen3-Omni** (released after the paper) is reported here for the first
  time, using the identical wrapper/code path.
- Full-set (61,320-sample) lmms-eval runs use the same wrappers via the
  Section 4 command; numbers are updated in
  [`lmms_eval/tasks/xmod_bench/RESULTS.md`](https://github.com/XingruiWang/lmms-eval/blob/feat/xmod-bench/lmms_eval/tasks/xmod_bench/RESULTS.md)
  as runs complete.

Per-run logs include overall accuracy plus per-config / per-family /
per-subtask breakdowns; `summarize.py` emits the 17 Level-2 numbers
(6 by-config, 5 by-family, 3 modality-disparity, 3 directional-imbalance).

## 📈 Benchmark Results

### Full Benchmark (61,320 samples) — all models from the paper

By-configuration accuracy (%) over the six modality directions; **Avg.** is the mean over the six (Table 2 of the paper).

| Model | A→T | A→V | T→A | T→V | V→A | V→T | Avg. |
|-------|----:|----:|----:|----:|----:|----:|-----:|
| Gemini 2.5 Pro | 71.0 | 58.9 | 64.4 | 79.8 | 60.8 | 88.6 | **70.6** |
| Gemini 2.5 Flash | 62.6 | 51.2 | 55.1 | 75.7 | 51.9 | 86.0 | 63.7 |
| Gemini 2.0 Flash | 63.7 | 49.0 | 52.2 | 71.5 | 47.6 | 85.2 | 61.2 |
| EchoInk-R1 | 64.6 | 45.9 | 56.4 | 60.9 | 49.9 | 77.6 | 59.2 |
| Qwen2.5-Omni | 62.0 | 48.0 | 55.4 | 59.6 | 50.5 | 76.3 | 58.6 |
| Gemini 1.5 Pro | 52.4 | 38.2 | 48.6 | 70.4 | 40.7 | 79.9 | 55.0 |
| Baichuan-Omni-1.5 | 47.8 | 35.8 | 40.5 | 56.2 | 38.6 | 73.0 | 48.7 |
| VideoLLaMA 2 | 48.6 | 26.0 | 25.7 | 26.5 | 25.2 | 66.8 | 36.5 |
| VITA | 40.2 | 26.0 | 29.8 | 26.8 | 29.9 | 59.3 | 35.4 |
| Unified-IO 2 XXL | 37.4 | 25.0 | 31.2 | 37.8 | 26.7 | 39.9 | 33.0 |
| Unified-IO 2 XL | 33.3 | 27.0 | 27.1 | 32.9 | 26.5 | 37.4 | 30.7 |
| Unified-IO 2 | 28.9 | 24.0 | 25.4 | 32.0 | 25.7 | 32.7 | 28.1 |
| PandaGPT | 24.5 | 25.0 | 23.8 | 25.2 | 24.5 | 25.1 | 24.7 |
| No Context (random) | 25.1 | 24.3 | 25.4 | 24.8 | 25.3 | 25.7 | 25.1 |
| _Human_ | 92.4 | 91.5 | 91.1 | 91.8 | 86.4 | 95.6 | _91.5_ |

> Vision-only models are evaluated only on text↔vision configs: **Qwen2.5-VL** 67.4 Avg., **InternVL-3.5** 61.7 Avg. (omitted from the six-way table).

### XModBench-Lite (6,000 samples) — reproduced via lmms-eval

Balanced split (5 families × 6 configs × 200), evaluated through the [lmms-eval port](https://github.com/XingruiWang/lmms-eval) with interleaved-multimedia wrappers.

| Model | A→T | A→V | T→A | T→V | V→A | V→T | Avg. |
|-------|----:|----:|----:|----:|----:|----:|-----:|
| Qwen3-Omni-30B | 71.6 | 52.0 | 62.5 | 67.0 | 55.6 | 83.1 | **65.3** |
| Qwen2.5-Omni-7B | 63.1 | 49.8 | 59.2 | 62.5 | 50.3 | 76.4 | 60.2 |
| Baichuan-Omni-1.5 | 52.5 | 32.0 | 47.6 | 56.6 | 47.0 | 77.7 | 52.2 |
| OmniVinci | 62.2 | — | — | — | — | 78.8 | — |

> Qwen2.5-Omni matches its full-set paper numbers within 5 points on every configuration, confirming the port is faithful. Qwen3-Omni post-dates the paper (first reported here). OmniVinci runs on its single-media-condition configs; its 4-option configs hit VILA-internal limits (see [`RESULTS.md`](lmms_eval/tasks/xmod_bench/RESULTS.md) — note: that file lives in the lmms-eval repo). New runs update as they complete.

### Key Findings

#### 1️⃣ Task Competence Gaps
- **Strong Performance**: Perception and linguistic tasks (~75% for best models)
- **Weak Performance**: Spatial (50.1%) and temporal reasoning (60.8%)
- **Performance Drop**: 15-25 points decrease in spatial/temporal vs. perception tasks

#### 2️⃣ Modality Disparity
- **Audio vs. Text**: 20-49 point performance drop
- **Audio vs. Vision**: 33-point average gap
- **Vision vs. Text**: ~15-point disparity
- **Consistency**: Best models show 10-12 point standard deviation

#### 3️⃣ Directional Imbalance
- **Vision↔Text**: 9-17 point gaps between directions
- **Audio↔Text**: 6-8 point asymmetries
- **Root Cause**: Training data imbalance favoring image-to-text over inverse directions

## 📝 Citation

If you use XModBench in your research, please cite our paper:

```bibtex
@article{wang2025xmodbench,
  title={XModBench: Benchmarking Cross-Modal Capabilities and Consistency in Omni-Language Models},
  author={Wang, Xingrui and others},
  journal={arXiv preprint arXiv:2510.15148},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank all contributors and the research community for their valuable feedback and suggestions.

## 📧 Contact

- **Project Lead**: Xingrui Wang
- **Email**: [xwang378@jhu.edu]
- **Website**: [https://xingruiwang.github.io/projects/XModBench/](https://xingruiwang.github.io/projects/XModBench/)

## 🔗 Links

- [Project Website](https://xingruiwang.github.io/projects/XModBench/)
- [Paper](https://arxiv.org/abs/xxxx.xxxxx)
- [Leaderboard](https://xingruiwang.github.io/projects/XModBench/leaderboard)
- [Documentation](https://xingruiwang.github.io/projects/XModBench/docs)


## Todo

- [x] Release Huggingface data
- [x] Release data processing code
- [x] Release data evaluation code
---

**Note**: XModBench is actively maintained and regularly updated with new models and evaluation metrics. For the latest updates, please check our [releases](https://github.com/XingruiWang/XModBench/releases) page.

