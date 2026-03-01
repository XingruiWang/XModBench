#! /bin/bash

export audioBench='/home/xwang378/scratch/2025/AudioBench'

# Configuration
MODEL="qwen2.5_omni"
# MODEL="vita"
# MODEL="echoink"
# MODEL="anygpt"
# MODEL="gemini-2.5-pro"
# MODEL="omnivinci"
# MODEL="qwen2.5_vl"
# MODEL="internvl3"
# MODEL="random"
# MODEL="panda"
# MODEL="reka"
# MINI_BENCHMARK="false"
MINI_BENCHMARK="true"

# Time logging setup
SCRIPT_START_TIME=$(date +%s)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$audioBench/logs/timing_${MODEL}_${TIMESTAMP}.log"
mkdir -p "$audioBench/logs"

# Initialize log file
echo "=== AudioBench Timing Log ===" > "$LOG_FILE"
echo "Model: $MODEL" >> "$LOG_FILE"
echo "Mini Benchmark: $MINI_BENCHMARK" >> "$LOG_FILE"
echo "Script Started: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Model initialization timing
MODEL_INIT_START_TIME=0
MODEL_INIT_END_TIME=0
EVALUATION_START_TIME=0

# Function to convert seconds to human readable format
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    
    if [ $hours -gt 0 ]; then
        printf "%02d:%02d:%02d" $hours $minutes $secs
    else
        printf "%02d:%02d" $minutes $secs
    fi
}

# Function to log timing information
log_timing() {
    local message="$1"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $message" | tee -a "$LOG_FILE"
}

# Function to warm up model (exclude init time from evaluation)
warm_up_model() {
    local model=$1
    
    echo "Warming up model: $model"
    log_timing "Starting model warm-up for: $model"
    
    MODEL_INIT_START_TIME=$(date +%s)
    
    # Use a simple task for warm-up
    if [ "$MINI_BENCHMARK" = "true" ]; then
        python $audioBench/scripts/run.py \
            --model $model \
            --task_name "perception/general_audio_vision" \
            --sample 1 \
            --mini_benchmark \
            > /dev/null 2>&1
    else
        python $audioBench/scripts/run.py \
            --model $model \
            --task_name "perception/general_audio_vision" \
            --sample 1 \
            > /dev/null 2>&1
    fi
    
    MODEL_INIT_END_TIME=$(date +%s)
    local init_duration=$((MODEL_INIT_END_TIME - MODEL_INIT_START_TIME))
    local init_formatted=$(format_duration $init_duration)
    
    log_timing "Model initialization completed in: ${init_formatted} (${init_duration}s)"
    echo "Model warm-up completed in: ${init_formatted}"
    echo ""
}

# Function to run evaluation
run_evaluation() {
    local model=$1
    local task=$2
    local subtask=$3
    local modality=$4
    
    local task_name="${task}/${subtask}_${modality}"
    
    echo "  Running modality: ${modality}"
    log_timing "    Starting modality: ${modality} for task ${task}/${subtask}"
    
    local modality_start_time=$(date +%s)
    
    if [ "$MINI_BENCHMARK" = "true" ]; then
        python $audioBench/scripts/run.py \
            --model $model \
            --task_name $task_name \
            --sample 3 \
            --mini_benchmark 
            # --reason True
    else
        python $audioBench/scripts/run.py \
            --model $model \
            --task_name $task_name \
            --sample 1000 
    fi
    
    local modality_end_time=$(date +%s)
    local modality_duration=$((modality_end_time - modality_start_time))
    local modality_formatted=$(format_duration $modality_duration)
    
    log_timing "    Completed modality: ${modality} in ${modality_formatted} (${modality_duration}s)"
}

# Main execution - uncomment the tasks you want to run
TASKS_TO_RUN=(
    # Perception tasks
    "perception/general"
    "perception/finegrained"
    "perception/instruments"
    "perception/instruments_comp"
    "perception/natures"
    
    # # Spatial tasks
    "spatial/arrangements"
    "spatial/3D_movements"
    "spatial/panaroma"
    
    # Speech tasks
    "speech/recognition"
    "speech/translation"
    
    # # Temporal tasks
    "temporal/count"
    "temporal/calculation"
    "temporal/order"
    
    # # External tasks
    "external/music_genre_classification"
    "external/emotion_classification"
    "external/movie_matching"
    "external/singer_identification"
)

# Warm up model first (exclude initialization time from evaluation timing)
warm_up_model $MODEL

# Start evaluation timing after model initialization
EVALUATION_START_TIME=$(date +%s)

# Run evaluations
task_counter=0
total_tasks=${#TASKS_TO_RUN[@]}

log_timing "Starting evaluation for $total_tasks tasks (model initialization excluded)"
echo "" >> "$LOG_FILE"

for task_key in "${TASKS_TO_RUN[@]}"; do
    task_counter=$((task_counter + 1))
    task=$(echo $task_key | cut -d'/' -f1)
    subtask=$(echo $task_key | cut -d'/' -f2)
    modalities="audio_vision vision_audio vision_text text_audio audio_text text_vision"
    # modalities="vision_text text_vision"
    # modalities="audio_vision"
    # modalities="audio_vision vision_audio vision_text text_vision"
    # modalities="text_vision"
    # modalities="audio_vision_text"
    
    # Count actual modalities to process
    modality_count=$(echo $modalities | wc -w)
    
    echo "Processing task: $task_key ($task_counter/$total_tasks)"
    echo "Modalities to process: $modality_count ($modalities)"
    echo "----------------------------------------"
    
    # Record task start time
    task_start_time=$(date +%s)
    log_timing "TASK START: $task_key ($task_counter/$total_tasks) - Processing $modality_count modalities"
    
    # Run all modalities for this task
    modality_counter=0
    for modality in $modalities; do
        modality_counter=$((modality_counter + 1))
        echo "  [$modality_counter/$modality_count]"
        run_evaluation $MODEL $task $subtask $modality 
    done
    
    # Record task end time and calculate duration
    task_end_time=$(date +%s)
    task_duration=$((task_end_time - task_start_time))
    task_formatted=$(format_duration $task_duration)
    
    echo "Completed task: $task_key"
    log_timing "TASK COMPLETED: $task_key - Duration: ${task_formatted} (${task_duration}s) - All $modality_count modalities finished"
    
    # Calculate and show progress (excluding model init time)
    evaluation_elapsed=$((task_end_time - EVALUATION_START_TIME))
    evaluation_elapsed_formatted=$(format_duration $evaluation_elapsed)
    
    echo "Progress: $task_counter/$total_tasks tasks completed"
    log_timing "PROGRESS: $task_counter/$total_tasks tasks completed - Evaluation time elapsed: ${evaluation_elapsed_formatted}"
    echo "" >> "$LOG_FILE"
    echo "========================================"
done

# Final timing summary
script_end_time=$(date +%s)
total_script_duration=$((script_end_time - SCRIPT_START_TIME))
total_formatted=$(format_duration $total_script_duration)

# Calculate pure evaluation time (excluding model initialization)
evaluation_duration=$((script_end_time - EVALUATION_START_TIME))
evaluation_formatted=$(format_duration $evaluation_duration)

# Calculate model initialization time
init_duration=$((MODEL_INIT_END_TIME - MODEL_INIT_START_TIME))
init_formatted=$(format_duration $init_duration)

echo "All evaluations completed!"
log_timing "ALL EVALUATIONS COMPLETED - Pure evaluation time: ${evaluation_formatted} (${evaluation_duration}s)"
echo "" >> "$LOG_FILE"
echo "=== FINAL SUMMARY ===" >> "$LOG_FILE"
echo "Model: $MODEL" >> "$LOG_FILE"
echo "Total tasks processed: $total_tasks" >> "$LOG_FILE"
echo "Model initialization time: ${init_formatted} (${init_duration}s)" >> "$LOG_FILE"
echo "Pure evaluation time: ${evaluation_formatted} (${evaluation_duration}s)" >> "$LOG_FILE"
echo "Total script duration: ${total_formatted} (${total_script_duration}s)" >> "$LOG_FILE"
echo "Average time per task (excluding init): $(format_duration $((evaluation_duration / total_tasks))) per task" >> "$LOG_FILE"
echo "Script completed: $(date)" >> "$LOG_FILE"
echo "Log saved to: $LOG_FILE"

# Display summary on console
echo ""
echo "=== TIMING SUMMARY ==="
echo "Model initialization: ${init_formatted}"
echo "Pure evaluation time: ${evaluation_formatted}"
echo "Total script time: ${total_formatted}"
echo "Average per task (excluding init): $(format_duration $((evaluation_duration / total_tasks)))"