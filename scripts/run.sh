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

python $audioBench/scripts/run.py \
    --model gemini \
    --task_name perception/vggss_vision_text \
    --sample 1000

