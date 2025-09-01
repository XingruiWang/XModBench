#!/bin/bash
#SBATCH --job-name=VLM_eval        
#SBATCH --output=log/job_%j.out
#SBATCH --error=log/job_%j.log    
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=1  
#SBATCH --exclude=ccvl37                                

echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Load user environment
source ~/.bashrc

module load conda
conda activate vlm

export audioBench='/home/xwang378/scratch/2025/AudioBench'

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name perception/vggss_audio_vision \
#     --sample -1


# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name perception/vggss_vision_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name perception/vggss_vision_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name perception/vggss_audio_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name perception/vggss_text_vision \
#     --sample -1


# python $audioBench/scripts/run.py \
#     --model gemini \
#     --task_name perception/vggss_text_audio \
#     --sample -1

######################## SOLOS ########################


# python $audioBench/scripts/run.py \
#     --model gemini-2.5-flash \
#     --task_name perception/solos_vision_audio \
#     --sample 970

# python $audioBench/scripts/run.py \
#     --model gemini-2.5-flash \
#     --task_name perception/solos_audio_vision \
#     --sample 970

# python $audioBench/scripts/run.py \
#     --model gemini-2.5-flash \
#     --task_name perception/solos_vision_text \
#     --sample 970

# python $audioBench/scripts/run.py \
#     --model gemini-2.5-flash \
#     --task_name perception/solos_text_vision \
#     --sample 970

# python $audioBench/scripts/run.py \
#     --model gemini-2.5-flash \
#     --task_name perception/solos_text_audio \
#     --sample 970

# python $audioBench/scripts/run.py \
#     --model gemini-2.5-flash \
#     --task_name perception/solos_audio_text \
#     --sample 970

######################## URMP ########################

# python $audioBench/scripts/run.py \
#     --model gemini \
#     --task_name perception/urmp_vision_audio \
#     --sample 200

# python $audioBench/scripts/run.py \
#     --model gemini \
#     --task_name perception/urmp_audio_vision \
#     --sample 200

# python $audioBench/scripts/run.py \
#     --model gemini \
#     --task_name perception/urmp_audio_text \
#     --sample 200

# python $audioBench/scripts/run.py \
#     --model gemini \
#     --task_name perception/urmp_text_audio \
#     --sample 200

# python $audioBench/scripts/run.py \
#     --model gemini \
#     --task_name perception/urmp_text_vision \
#     --sample 200

# python $audioBench/scripts/run.py \
#     --model gemini \
#     --task_name perception/urmp_vision_text \
#     --sample 200


########################## Landscapes ########################

# python $audioBench/scripts/run.py \
#     --model gemini_2.0_flash \
#     --task_name perception/landscapes_audio_vision \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model gemini_2.0_flash \
#     --task_name perception/landscapes_vision_audio \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name perception/landscapes_audio_text \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name perception/landscapes_text_audio \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name perception/landscapes_vision_text \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name perception/landscapes_text_vision \
#     --sample 500


########################## ACR ########################

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name acr/acr_audio_vision \
#     --sample 970

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name acr/acr_vision_audio \
#     --sample 970


# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name acr/acr_audio_text \
#     --sample 970


# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name acr/acr_text_audio \
#     --sample 970


# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name acr/acr_vision_text \
#     --sample 970



# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name acr/acr_text_vision \
#     --sample 970

########################## OCR Hard ########################


# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name acr/acr_hard_audio_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name acr/acr_hard_vision_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name acr/acr_hard_text_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name acr/acr_hard_vision_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name acr/acr_hard_text_vision \
#     --sample -1


# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name acr/acr_hard_audio_vision \
#     --sample -1




########################## Spatial ########################

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_easy/urmp_audio_vision \
#     --sample 500


# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_easy/urmp_vision_audio \
#     --sample 500


# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_easy/urmp_audio_text \
#     --sample 500


# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_easy/urmp_text_audio \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_easy/urmp_text_vision \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_easy/urmp_vision_text \
#     --sample 500

########################## OCR Translation ########################

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name ocr_translation_Chinese/ocr_translation_audio_text \
#     --sample 970

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name ocr_translation_Chinese/ocr_translation_text_audio \
#     --sample 970

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name ocr_translation_Chinese/ocr_translation_vision_text \
#     --sample 970

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name ocr_translation_Chinese/ocr_translation_text_vision \
#     --sample 970

########################## OCR Translation Hard ########################

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name acr_translation_Chinese_hard/ocr_translation_text_audio \
#     --sample 970

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name acr_translation_Chinese_hard/ocr_translation_vision_text \
#     --sample 970

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name acr_translation_Chinese_hard/ocr_translation_vision_audio \
#     --sample 970

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name acr_translation_Chinese_hard/ocr_translation_audio_text \
#     --sample 970


########################## Count ########################

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count/countixav_audio_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count/countixav_text_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count/countixav_vision_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count/countixav_text_vision \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count/countixav_audio_vision \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count/countixav_vision_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count/countixav_video_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count/countixav_text_video \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count/countixav_audio_video \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count/countixav_video_audio \
#     --sample -1

########################### Count Reasoning ########################
# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count_reasoning/countixav_vision_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count_reasoning/countixav_audio_vision \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count_reasoning/countixav_vision_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count_reasoning/countixav_text_vision \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count_reasoning/countixav_text_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count_reasoning/countixav_audio_text \
#     --sample -1


# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count_reasoning/countixav_video_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count_reasoning/countixav_audio_video \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count_reasoning/countixav_video_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count_reasoning/countixav_text_video \
#     --sample -1

################ Temporal Order ################ 
# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_order/vggss_order_audio_text \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_order/vggss_order_text_audio \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_order/vggss_order_vision_text \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_order/vggss_order_text_vision \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_order/vggss_order_vision_audio \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_order/vggss_order_audio_vision \
#     --sample 500

################ Spatial Outdoor ################

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_outdoor/urbansas_audio_vision \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_outdoor/urbansas_vision_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_outdoor/urbansas_audio_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_outdoor/urbansas_text_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_outdoor/urbansas_vision_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_outdoor/urbansas_text_vision \
#     --sample -1

################ Spatial Indoor ################

# Audio to Video
# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_indoor/starss23_audio_video \
#     --sample 97

# # Video to Audio  
# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_indoor/starss23_video_audio \
#     --sample 97

# # Audio to Text
# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_indoor/starss23_audio_text \
#     --sample 97

# # Text to Audio
# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_indoor/starss23_text_audio \
#     --sample 97

# # Video to Text
# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_indoor/starss23_video_text \
#     --sample 97

# # Text to Video
# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_indoor/starss23_text_video \
#     --sample 97


########################## External ########################

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/music_genre_classification_audio_vision \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/music_genre_classification_vision_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/music_genre_classification_audio_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/music_genre_classification_text_audio \
#     --sample -1


# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/music_genre_classification_vision_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/music_genre_classification_text_vision \
#     --sample -1

########################## Emotion ########################

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/emotion_classification_audio_vision \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/emotion_classification_vision_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/emotion_classification_audio_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/emotion_classification_text_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/emotion_classification_vision_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/emotion_classification_text_vision \
#     --sample -1

########################## Movie Matching ########################

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/movie_matching_audio_vision \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/movie_matching_vision_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/movie_matching_audio_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/movie_matching_text_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/movie_matching_vision_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/movie_matching_text_vision \
#     --sample -1

########################## Singer Identification ########################

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/singer_identification_audio_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/singer_identification_text_audio \
#     --sample -1


# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/singer_identification_vision_text \
#     --sample -1


# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name external/singer_identification_vision_audio \
#     --sample -1


python $audioBench/scripts/run.py \
    --model gemini-2.0-flash \
    --task_name external/singer_identification_audio_vision \
    --sample -1

python $audioBench/scripts/run.py \
    --model gemini-2.0-flash \
    --task_name external/singer_identification_text_vision \
    --sample -1





