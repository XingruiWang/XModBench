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


######################## Qwen2.5-Omni; VGGSS ########################

# python $audioBench/scripts/run.py \
#         --model qwen2.5_omni \
#         --task_name perception/vggss_audio_text \
#         --sample 1000

# python $audioBench/scripts/run.py \
#         --model qwen2.5_omni \
#         --task_name perception/vggss_vision_text \
#         --sample 1000

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/vggss_text_vision \
#     --sample 1000


# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/vggss_text_audio \
#     --sample 1000

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/vggss_vision_audio \
#     --sample 1000

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/vggss_audio_vision \
#     --sample 1000

######################## Qwen2.5-Omni; SOLOS ########################


# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/solos_vision_audio \
#     --sample 1000

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/solos_audio_vision \
#     --sample 1000

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/solos_vision_text \
#     --sample 1000

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/solos_audio_text \
#     --sample 1000

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/solos_text_vision \
#     --sample 1000

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/solos_text_audio \
#     --sample 1000


######################## Qwen2.5-Omni; URMP ########################

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/urmp_vision_audio \
#     --sample 200

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/urmp_audio_vision \
#     --sample 200

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/urmp_audio_text \
#     --sample 200

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/urmp_text_audio \
#     --sample 200

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/urmp_text_vision \
#     --sample 200

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/urmp_vision_text \
#     --sample 200

########################## Landscapes ########################

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/landscapes_audio_vision \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/landscapes_vision_audio \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/landscapes_audio_text \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/landscapes_text_audio \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/landscapes_text_vision \
#     --sample 500

python $audioBench/scripts/run.py \
    --model qwen2.5_omni \
    --task_name perception/landscapes_vision_text \
    --sample 500