export audioBench='/home/xwang378/scratch/2025/AudioBench'


# python $audioBench/scripts/run.py \
#     --model gemini \
#     --task_name perception/vggss_audio_vision \
#     --sample 1

# python $audioBench/scripts/run.py \
#     --model gemini \
#     --task_name perception/vggss_vision_audio \
#     --sample 100  

# python $audioBench/scripts/run.py \
#     --model gemini \
#     --task_name perception/vggss_audio_text \
#     --sample 10

# python $audioBench/scripts/run.py \
#     --model gemini \
#     --task_name spatial/urmp_vision_text \
#     --sample 10

# python $audioBench/scripts/run.py \
#     --model gemini \
#     --task_name acr/acr_audio_vision \
#     --sample 10

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name spatial_easy/urmp_vision_text \
#     --sample 500

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name acr/acr_hard_text_audio  \
#     --sample 100

# python $audioBench/scripts/run.py \
#     --model qwen2.5_omni \
#     --task_name acr_translation_Chinese_hard/ocr_translation_audio_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count/countixav_video_text \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count_reasoning/countixav_vision_audio \
#     --sample -1

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count_reasoning/countixav_vision_audio   \
#     --sample 10 \
#     --reason True

# python $audioBench/scripts/run.py \
#     --model gemini-2.0-flash \
#     --task_name temporal_count/countixav_audio_text  \
#     --sample 10 \
#     --reason True

# Text to Audio
python $audioBench/scripts/run.py \
    --model gemini-2.0-flash \
    --task_name spatial_indoor/starss23_text_video \
    --sample 100