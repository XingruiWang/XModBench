# XModBench: Benchmarking Cross-Modal Capabilities and Consistency in Omni-Language Models

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2510.15148)
[![Website](https://img.shields.io/badge/Website-XModBench-green.svg)](https://xingruiwang.github.io/projects/XModBench/)
[![Dataset](https://img.shields.io/badge/Dataset-XModBench-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/RyanWW/XModBench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


XModBench is a comprehensive benchmark designed to evaluate the cross-modal capabilities and consistency of omni-language models. It systematically assesses model performance across multiple modalities (text, vision, audio) and various cognitive tasks, revealing critical gaps in current state-of-the-art models.

### Key Features

- **🎯 Multi-Modal Evaluation**: Comprehensive testing across text, vision, and audio modalities
- **🧩 5 Task Dimensions**: Perception, Spatial, Temporal, Linguistic, and Knowledge tasks
- **📊 13 SOTA Models Evaluated**: Including Gemini 2.5 Pro, Qwen2.5-Omni, EchoInk-R1, and more
- **🔄 Consistency Analysis**: Measures performance stability across different modal configurations
- **👥 Human Performance Baseline**: Establishes human-level benchmarks for comparison


## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/XingruiWang/XModBench.git
cd XModBench

# Install dependencies
pip install -r requirements.txt
```

## 📂 Dataset Structure

```
XModBench/
├── data/
│   ├── text/
│   │   ├── perception/
│   │   ├── spatial/
│   │   ├── temporal/
│   │   ├── linguistic/
│   │   └── knowledge/
│   ├── vision/
│   │   └── [same task categories]
│   └── audio/
│       └── [same task categories]
├── models/
│   └── evaluation_scripts/
├── results/
│   └── model_performances/
└── analysis/
    └── visualization/
```



### Basic Usage

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
@article{wang2024xmodbench,
  title={XModBench: Benchmarking Cross-Modal Capabilities and Consistency in Omni-Language Models},
  author={Wang, Xingrui and Others},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank all contributors and the research community for their valuable feedback and suggestions.

## 📧 Contact

- **Project Lead**: Xingrui Wang
- **Email**: [xingrui.wang@example.edu]
- **Website**: [https://xingruiwang.github.io/projects/XModBench/](https://xingruiwang.github.io/projects/XModBench/)

## 🔗 Links

- [Project Website](https://xingruiwang.github.io/projects/XModBench/)
- [Paper](https://arxiv.org/abs/xxxx.xxxxx)
- [Leaderboard](https://xingruiwang.github.io/projects/XModBench/leaderboard)
- [Documentation](https://xingruiwang.github.io/projects/XModBench/docs)


## Todo

- [ ] Release Huggingface data
- [x] Release data processing code
- [x] Release data evaluation code
---

**Note**: XModBench is actively maintained and regularly updated with new models and evaluation metrics. For the latest updates, please check our [releases](https://github.com/XingruiWang/XModBench/releases) page.

