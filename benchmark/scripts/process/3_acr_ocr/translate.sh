export audioBench='/home/xwang378/scratch/2025/AudioBench'

python generate_translation.py \
    --ocr_bench_path /home/xwang378/scratch/2025/AudioBench/benchmark/Data/rendertext/audio \
    --language Chinese \
    --model gemini-2.0-flash \
    --output_path /home/xwang378/scratch/2025/AudioBench/benchmark/Data/rendertext/audio/translation

python generate_translation.py \
    --ocr_bench_path /home/xwang378/scratch/2025/AudioBench/benchmark/Data/rendertext/audio \
    --language Japanese \
    --model gemini-2.0-flash \
    --output_path /home/xwang378/scratch/2025/AudioBench/benchmark/Data/rendertext/audio/translation

python generate_translation.py \
    --ocr_bench_path /home/xwang378/scratch/2025/AudioBench/benchmark/Data/rendertext/audio \
    --language Spanish \
    --model gemini-2.0-flash \
    --output_path /home/xwang378/scratch/2025/AudioBench/benchmark/Data/rendertext/audio/translation
    