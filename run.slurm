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

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name perception/landscapes_vision_text \
#     --sample 500

########################## ACR ########################

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name acr/acr_audio_vision \
#     --sample 1000

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name acr/acr_vision_audio \
#     --sample 1000


# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name acr/acr_audio_text \
#     --sample 1000


# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name acr/acr_text_audio \
#     --sample 1000


# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name acr/acr_vision_text \
#     --sample 1000



# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name acr/acr_text_vision \
#     --sample 1000

########################## Spatial ########################

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name spatial/urmp_audio_vision \
#     --sample 500


# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name spatial/urmp_vision_audio \
#     --sample 500


# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name spatial/urmp_audio_text \
#     --sample 500


# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name spatial/urmp_text_audio \
#     --sample 500


# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name spatial/urmp_vision_text \
#     --sample 500


# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name spatial/urmp_text_vision \
#     --sample 500

########################## OCR Translation ########################

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name ocr_translation_Chinese/ocr_translation_audio_text \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name ocr_translation_Chinese/ocr_translation_text_audio \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name ocr_translation_Chinese/ocr_translation_text_vision \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name ocr_translation_Chinese/ocr_translation_vision_text \
#     --sample 500

########################## ACR Hard ########################

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name acr/acr_hard_audio_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name acr/acr_hard_vision_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name acr/acr_hard_text_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name acr/acr_hard_vision_audio \
#     --sample -1

########################## OCR Translation Hard ########################

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name acr_translation_Chinese_hard/ocr_translation_text_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name acr_translation_Chinese_hard/ocr_translation_vision_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name acr_translation_Chinese_hard/ocr_translation_vision_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name acr_translation_Chinese_hard/ocr_translation_audio_text \
#     --sample -1



################ Temporal Order ################ 
# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name temporal_order/vggss_order_audio_text \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name temporal_order/vggss_order_text_audio \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name temporal_order/vggss_order_vision_text \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name temporal_order/vggss_order_text_vision \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name temporal_order/vggss_order_vision_audio \
#     --sample 500

python $audioBench/scripts/run.py \
    --model qwen2.5_omni \
    --task_name temporal_order/vggss_order_audio_vision \
    --sample 49

# ############### Spatial Outdoor ################

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name spatial_outdoor/urbansas_audio_vision \
#     --sample 98

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name spatial_outdoor/urbansas_vision_audio \
#     --sample 98

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name spatial_outdoor/urbansas_audio_text \
#     --sample 98

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name spatial_outdoor/urbansas_text_audio \
#     --sample 98

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name spatial_outdoor/urbansas_vision_text \
#     --sample 98

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name spatial_outdoor/urbansas_text_vision \
#     --sample 98


########################## Count ########################

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name temporal_count/countixav_audio_text \
#     --sample 47

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name temporal_count/countixav_text_audio \
#     --sample 47

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name temporal_count/countixav_vision_text \
#     --sample 47

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name temporal_count/countixav_text_vision \
#     --sample 47

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name temporal_count/countixav_audio_vision \
#     --sample 47

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name temporal_count/countixav_vision_audio \
#     --sample 47

########################### Count Reasoning ########################
# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name temporal_count_reasoning/countixav_vision_audio \
#     --sample 47

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name temporal_count_reasoning/countixav_audio_vision \
#     --sample 47

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name temporal_count_reasoning/countixav_vision_text \
#     --sample 47

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name temporal_count_reasoning/countixav_text_vision \
#     --sample 47

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name temporal_count_reasoning/countixav_text_audio \
#     --sample 47

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name temporal_count_reasoning/countixav_audio_text \
#     --sample 47




