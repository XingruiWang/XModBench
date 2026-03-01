#!/bin/bash
#SBATCH --job-name=vlm_eval  
#SBATCH --output=log/job_%j.out
#SBATCH --error=log/job_%j.log                        
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --partition=main
#SBATCH --mem=32G


# 在批处理环境中正确初始化conda
module load conda

# 直接source conda的初始化脚本而不依赖.bashrc
source /home/apps/anaconda3/etc/profile.d/conda.sh

# 验证conda可用
which conda
conda --version

# 激活环境
conda activate /home/xwang378/.conda/envs/qwenomni3

# 验证环境激活成功
echo "Active environment: $CONDA_DEFAULT_ENV"
which python 
# conda activate /home/xwang378/.conda/envs/vita
# conda activate /home/xwang378/.conda/envs/AnyGPT
# conda activate /home/xwang378/.conda/envs/panda
# conda activate /home/xwang378/.conda/envs/vlm
 
echo "Running on host: $(hostname)"
# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

export audioBench='/home/xwang378/scratch/2025/AudioBench'

# Configuration
# MODEL="qwen3-omni"
# MODEL="omnivinci"
# MODEL="qwen2.5_vl"
MODEL="internvl3"
# MODEL="vita"
# MODEL="gemini-1.5-pro"
# MODEL="gemini-2.0-flash"
# MODEL="gemini-2.5-pro"
# MODEL="anygpt"
# MODEL="panda"
# MODEL="echoink"
# MINI_BENCHMARK="false"
MINI_BENCHMARK="true"

# Function to run evaluation
run_evaluation() {
    local model=$1
    local task=$2
    local subtask=$3
    local modality=$4
    
    local task_name="${task}/${subtask}_${modality}"
    
    echo "Running: Model=${model}, Task=${task_name}"
    
    if [ "$MINI_BENCHMARK" = "true" ]; then
        python $audioBench/scripts/run.py \
            --model $model \
            --task_name $task_name \
            --sample 297 \
            --mini_benchmark
    else
        python $audioBench/scripts/run.py \
            --model $model \
            --task_name $task_name \
            --sample 1000 \
            --reason True
    fi
}

modalities="audio_vision vision_audio vision_text text_vision text_audio audio_text"
# modalities="vision_text text_vision"
# modalities="text_audio audio_text"
# modalities="text_audio"
# modalities="audio_text"
# modalities="audio_vision_text"
# modalities="vision_text"
# modalities="audio_text text_vision vision_text"
# modalities="audio_text text_vision text_audio"
# modalities="vision_text"

# Main execution - uncomment the tasks you want to run
TASKS_TO_RUN=(
    # Perception tasks
    "perception/general"
    "perception/finegrained"
    # "perception/instruments"
    # "perception/instruments_comp"
    # "perception/natures"
    
    # # Spatial tasks
    # "spatial/arrangements"
    # "spatial/3D_movements"
    # "spatial/panaroma"
    
    # Speech tasks
    # "speech/recognition"
    # "speech/translation"
    
    # # Temporal tasks
    # "temporal/count"
    # "temporal/calculation"
    # "temporal/order"
    
    # # External tasks
    # "external/music_genre_classification"
    # "external/emotion_classification"
    # "external/movie_matching"
    # "external/singer_identification"
)

# Run evaluations
for task_key in "${TASKS_TO_RUN[@]}"; do
    task=$(echo $task_key | cut -d'/' -f1)
    subtask=$(echo $task_key | cut -d'/' -f2)
    
    echo "Processing task: $task_key"
    echo "Modalities: $modalities"
    echo "----------------------------------------"
    
    for modality in $modalities; do
        run_evaluation $MODEL $task $subtask $modality 
    done
    
    echo "Completed task: $task_key"
    echo "========================================"
done

echo "All evaluations completed!"