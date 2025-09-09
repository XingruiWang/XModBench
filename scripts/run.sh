#! /bin/bash

export audioBench='/home/xwang378/scratch/2025/AudioBench'

# Configuration
# MODEL="qwen2.5_omni"
MODEL="vita"
# MODEL="echoink"
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
        CUDA_VISIBLE_DEVICES=0 python $audioBench/scripts/run.py \
            --model $model \
            --task_name $task_name \
            --sample 97 \
            --mini_benchmark
    else
        CUDA_VISIBLE_DEVICES=0 python $audioBench/scripts/run.py \
            --model $model \
            --task_name $task_name
    fi
}

# Main execution - uncomment the tasks you want to run
TASKS_TO_RUN=(
    # # Perception tasks
    "perception/general"
    "perception/finegrained"
    "perception/instruments"
    "perception/instruments_comp"
    "perception/natures"
    
    # Spatial tasks
    "spatial/arrangements"
    "spatial/3D_movements"
    "spatial/panaroma"
    
    # Speech tasks
    "speech/recognition"
    "speech/translation"
    
    # Temporal tasks
    "temporal/count"
    "temporal/calculation"
    "temporal/order"
    
    # External tasks
    "external/music_genre_classification"
    "external/emotion_classification"
    "external/movie_matching"
    "external/singer_identification"
)

# Run evaluations
for task_key in "${TASKS_TO_RUN[@]}"; do
    task=$(echo $task_key | cut -d'/' -f1)
    subtask=$(echo $task_key | cut -d'/' -f2)
    modalities="audio_vision vision_audio vision_text text_vision text_audio audio_text"
    
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