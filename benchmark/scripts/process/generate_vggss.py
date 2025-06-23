import os
import re
import random
import torch
import soundfile as sf
import json

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

print("Available GPUs:", torch.cuda.device_count())



# ==================== 1. Main function: Load model and answer one question ====================




def generate_question_vision(audio_choices, audio_paths, correct_answer):
    
    image_frames_folders = [audio_sample.replace('.wav', '_frames') for audio_sample in audio_paths]
    image_paths = [os.path.join(folder, os.listdir(folder)[0]) for folder in image_frames_folders]
    
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    
    question = {
        "question": "Which image most likely belongs to the object that make this sound? Answer the question with A, B, C, or D",
        "conditions": {
            "modality": "Audio",
            "input": correct_audio_sample,
            },
        "options": {
            "A": {
                "modality": "Image",
                "input": image_paths[0],
            },
            "B": {
                "modality": "Image",
                "input": image_paths[1],
            },
            "C": {
                "modality": "Image",
                "input": image_paths[2],
            },
            "D": {
                "modality": "Image",
                "input": image_paths[3],
            }
        },
        "correct_answer": correct_answer
    }
    
    return conversation, question


def generate_question_audio(audio_choices, audio_paths, correct_answer):
    
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    
    image_frames_folder = correct_audio_sample.replace('.wav', '_frames')
    image_path = os.path.join(image_frames_folder, os.listdir(image_frames_folder)[0])  # Use the first frame as the image
    
    question = {
        "question": "Which sound is most likely made by the object from this image? Answer the question with A, B, C, or D",
        "conditions": {
            "modality": "Image",
            "input": image_path,
            },
        "options": {
            "A": {
                "modality": "Audio",
                "input": audio_paths[0],
            },
            "B": {
                "modality": "Audio",
                "input": audio_paths[1],
            },
            "C": {
                "modality": "Audio",
                "input": audio_paths[2],
            },
            "D": {
                "modality": "Audio",
                "input": audio_paths[3],
            }
        },
        "correct_answer": correct_answer
    }
    
    return conversation, question


if __name__ == "__main__":
    random.seed(42)  # For reproducibility

    # Set fixed audio choices for all questions
    all_audio_choices = os.listdir("/home/xwang378/scratch/Data/vggss_audio_bench/")
    all_audio_choices = [audio_name for audio_name in all_audio_choices if audio_name.endswith('.wav')]
    
    # Directory of frames
    valid_all_audio_choices = []
    for audio_name in all_audio_choices:
        frames_folder = audio_name.replace('.wav', '_frames')
        frames_path = os.path.join("/home/xwang378/scratch/Data/vggss_audio_bench/", frames_folder)
        if not os.path.exists(frames_path):
            print(f"Frames folder does not exist for {audio_name}, skipping.")
            continue
        if len(os.listdir(frames_path)) == 0:
            print(f"No frames found in {frames_folder}, skipping.")
            continue
        valid_all_audio_choices.append(audio_name)
    all_audio_choices = valid_all_audio_choices
    
    print(f"Total valid audio choices: {len(all_audio_choices)}")
    
    
    
    correct_vision = 0
    correct_audio = 0
    all_sample = 0
    
    audio_questions = []
    vision_questions = []
    for i in range(1000):
        audio_choices = random.sample(all_audio_choices, 4)
        audio_paths = [os.path.join("/home/xwang378/scratch/Data/vggss_audio_bench/", audio_name) for audio_name in audio_choices]

        correct_answers = random.choice(['A', 'B', 'C', 'D'])
        
        question_vision = \
            generate_question_vision(audio_choices, audio_paths, correct_answers)
        
        question_audio = \
            generate_question_audio(audio_choices, audio_paths, correct_answers)

        
        audio_questions.append(question_audio)
        vision_questions.append(question_vision)

    
    with open("vggss_audio_bench_questions_audio.json", "w") as f:
        json.dump(audio_questions, f, indent=4)
    with open("vggss_audio_bench_questions_vision.json", "w") as f:
        json.dump(vision_questions, f, indent=4)
    