

import json
import os

model = "gemini-2.5-pro"
TASKS_TO_RUN=(
    # Perception tasks
    "perception/general",
    "perception/finegrained",
    "perception/instruments",
    "perception/instruments_comp",
    "perception/natures",
    
    # Spatial tasks
    "spatial/arrangements",
    "spatial/3D_movements",
    "spatial/panaroma",
    
    # Speech tasks
    "speech/recognition",
    "speech/translation",
    
    # Temporal tasks
    "temporal/count",
    "temporal/calculation",
    "temporal/order",
    
    # External tasks
    "external/music_genre_classification",
    "external/emotion_classification",
    "external/movie_matching",
    "external/singer_identification"
)

modalities=["audio_vision", "vision_audio", "vision_text", "text_audio", "audio_text", "text_vision"]

# save_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/results/gemini-2.5-pro"
save_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/results/qwen2.5_omni"



id_to_result = {}
for task in TASKS_TO_RUN:
    id_to_result[task] = {}
    result_modality = {}
    try:
        for modality in modalities:
            with open(os.path.join(save_dir, f"{task.replace('/', '_')}_{modality}.json"), "r") as f:
                results = json.load(f)
            results = results["results"]
            result_modality[modality] = results
    except:
        continue
    
    for index, result in enumerate(result_modality["audio_vision"]):
        audio_vision_result = result
        try:
            vision_audio_result = result_modality["vision_audio"][str(index)]
            vision_text_result = result_modality["vision_text"][str(index)]
            text_audio_result = result_modality["text_audio"][str(index)]
            audio_text_result = result_modality["audio_text"][str(index)]
            text_vision_result = result_modality["text_vision"][str(index)]
        except:
            continue
        
        incorrect_count = 0
        current_result = {}
        for modality in modalities:
            current_result[modality] = result_modality[modality][str(index)]
            if not result_modality[modality][str(index)]["is_correct"] and result_modality[modality][str(index)]["response"] in ["A", "B", "C", "D"]:
                incorrect_count += 1
        # if incorrect_count >= 1 and incorrect_count < len(modalities):
        if audio_text_result["is_correct"] and not text_audio_result["is_correct"]:
            print(task, index)
            id_to_result[task][index] = current_result

with open(os.path.join('/home/xwang378/scratch/2025/AudioBench/benchmark/results/qwen2.5_omni/', "hard_case.json"), "w") as f:
    json.dump(id_to_result, f, indent=4)