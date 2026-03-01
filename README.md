<h1 align="center">
XModBench: Benchmarking Cross-Modal Capabilities and Consistency in Omni-Language Models
</h1>

<p align="center">
  <img src="https://xingruiwang.github.io/projects/XModBench/static/images/teaser.png" width="90%" alt="XModBench teaser">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.15148">
    <img src="https://img.shields.io/badge/Arxiv-Paper-b31b1b.svg" alt="Paper">
  </a>
  <a href="https://xingruiwang.github.io/projects/XModBench/">
    <img src="https://img.shields.io/badge/Website-Page-0a7aca?logo=globe&logoColor=white" alt="Website">
  </a>
  <a href="https://huggingface.co/datasets/RyanWW/XModBench">
    <img src="https://img.shields.io/badge/Huggingface-Dataset-FFD21E?logo=huggingface" alt="Dataset">
  </a>
<a href="https://github.com/XingruiWang/XModBench">
  <img src="https://img.shields.io/badge/Github-Code-181717?logo=github&logoColor=white" alt="GitHub Repo">
</a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  </a>
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

| Group | Subtasks | Samples |
|---|---|---:|
| Perception | finegrained, general_activities, instruments, instruments_comp, natures | 27,000 |
| Spatial | 3D_movements, arrangements, panaroma | 7,791 |
| Speech | recognition, translation | 8,244 |
| Temporal | calculation, count, order | 9,000 |
| External Knowledge | emotion_classification, movie_matching, music_genre_classification, singer_identification | 12,300 |
| **Total** | **17 subtasks** | **64,335** |

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



### lmms-eval Evaluation (recommended for open-source models)

For systematic, reproducible evaluation of open-source omni-LMMs we use [**lmms-eval**](https://github.com/XingruiWang/lmms-eval), a fork of the lmms-eval framework with XModBench tasks pre-integrated.

> **Note:** Vision has been split into **Image** and **Video** for efficient evaluation — models only need to load the relevant media type per task.

#### 1. Clone and install lmms-eval

```bash
git clone https://github.com/XingruiWang/lmms-eval.git
cd lmms-eval
pip install -e ".[all]"
```

#### 2. Set the data root and generate JSONL files

```bash
export XMODBENCH=/path/to/XModBench

python lmms_eval/tasks/xmod_bench/build_data.py \
    --tasks-root $XMODBENCH/benchmark/tasks \
    --out-dir    lmms_eval/tasks/xmod_bench/data \
    --seed 42
```

This generates 10 JSONL files (one per modality combination) in `lmms_eval/tasks/xmod_bench/data/`.

#### 3. Run a quick test

```bash
python -m lmms_eval \
    --model qwen2_5_omni \
    --model_args pretrained=Qwen/Qwen2.5-Omni-7B \
    --tasks xmod_bench_image_text \
    --batch_size 1 \
    --limit 16
```

#### 4. Full benchmark with Slurm (all 10 modality combinations in parallel)

```bash
#!/bin/bash
#SBATCH --job-name=xmod_bench_qwen2_5_omni
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/xmod_bench/%x_%a.log
#SBATCH --error=logs/xmod_bench/%x_%a.log

TASKS=(
    xmod_bench_audio_text    # 10,720 samples
    xmod_bench_text_audio    # 10,725 samples
    xmod_bench_audio_image   #  7,689 samples
    xmod_bench_image_audio   #  7,689 samples
    xmod_bench_image_text    #  7,689 samples
    xmod_bench_text_image    #  7,689 samples
    xmod_bench_audio_video   #  3,031 samples
    xmod_bench_text_video    #  3,031 samples
    xmod_bench_video_audio   #  3,036 samples
    xmod_bench_video_text    #  3,036 samples
)

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
REPO=/path/to/lmms-eval
export XMODBENCH=/path/to/XModBench

cd "$REPO"
source .venv/bin/activate

python -m lmms_eval \
    --model qwen2_5_omni \
    --model_args pretrained=Qwen/Qwen2.5-Omni-7B \
    --tasks "$TASK" \
    --batch_size 1 \
    --output_path "$REPO/logs/xmod_bench/results" \
    --log_samples \
    --log_samples_suffix "$TASK"
```

Submit all 10 tasks at once:
```bash
sbatch run_xmod_bench.slurm
```

Evaluation results include overall accuracy and per-group / per-subtask / per-modality-combo breakdowns, logged automatically at the end of each run.

## 📈 Benchmark Results

### Overall Performance Comparison

| Model | Perception | Spatial | Temporal | Linguistic | Knowledge | Average |
|-------|------------|---------|----------|------------|-----------|---------|
| **Gemini 2.5 Pro** | 75.9% | 50.1% | 60.8% | 76.8% | 89.3% | 70.6% |
| **Human Performance** | 91.0% | 89.7% | 88.9% | 93.9% | 93.9% | 91.5% |

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

