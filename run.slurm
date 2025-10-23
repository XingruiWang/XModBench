#!/bin/bash
#SBATCH --job-name=cpu   
#SBATCH --output=log/job_%j.out
#SBATCH --error=log/job_%j.log                        
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --exclude=ccvl14,ccvl30


module load conda
conda activate /home/xwang378/.conda/envs/omni
# conda activate /home/xwang378/.conda/envs/vita
# conda activate /home/xwang378/.conda/envs/AnyGPT
# conda activate /home/xwang378/.conda/envs/panda
# conda activate /home/xwang378/.conda/envs/vlm
 
echo "Running on host: $(hostname)"
# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

export audioBench='/home/xwang378/scratch/2025/AudioBench'

# Configuration
MODEL="qwen2.5_omni"
# MODEL="vita"
# MODEL="gemini-1.5-pro"
# MODEL="gemini-2.0-flash"
# MODEL="gemini-2.5-pro"
# MODEL="anygpt"
# MODEL="panda"
# MODEL="echoink"
# MINI_BENCHMARK="false"
MINI_BENCHMARK="false"

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

# modalities="audio_vision vision_audio vision_text text_vision text_audio audio_text"
# modalities="text_audio audio_text"
# modalities="text_audio"
modalities="audio_text"
# modalities="audio_vision_text"
# modalities="vision_text"
# modalities="audio_text text_vision vision_text"
# modalities="audio_text text_vision text_audio"
# modalities="vision_text"

# Main execution - uncomment the tasks you want to run
TASKS_TO_RUN=(
    # Perception tasks
    # "perception/general"
    # "perception/finegrained"
    # "perception/instruments"
    # "perception/instruments_comp"
    # "perception/natures"
    
    # # Spatial tasks
    # "spatial/arrangements"
    "spatial/3D_movements"
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