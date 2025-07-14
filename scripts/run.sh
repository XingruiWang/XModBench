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
#     --task_name spatial/urmp_audio_vision \
#     --sample 100

python $audioBench/scripts/run.py \
    --model gemini-2.0-flash \
    --task_name acr/acr_hard_text_audio  \
    --sample 100