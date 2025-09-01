import os
import re
import random
import json


# ==================== 1. Main function: Load model and answer one question ====================


def generate_question_audio_vision(audio_choices, audio_paths, correct_answer, frame_in_folder=True, object_name='objects'):
    
    if frame_in_folder:
        image_frames_folders = [audio_sample.replace('.wav', '_frames') for audio_sample in audio_paths]
        image_paths = [os.path.join(folder, os.listdir(folder)[0]) for folder in image_frames_folders]
    else:
        image_paths = [audio_sample.replace('.wav', '.jpg') for audio_sample in audio_paths]    
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    
    question = {
        "question": f"Which image most likely belongs to the {object_name} that make this sound? Answer the question with A, B, C, or D",
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
    
    return question


def generate_question_vision_audio(audio_choices, audio_paths, correct_answer, frame_in_folder=True, object_name='objects'):
    
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    
    # image_frames_folder = correct_audio_sample.replace('.wav', '_frames')
    # image_path = os.path.join(image_frames_folder, os.listdir(image_frames_folder)[0])  # Use the first frame as the image
    if frame_in_folder:
        image_frames_folder = correct_audio_sample.replace('.wav', '_frames')
        image_path = os.path.join(image_frames_folder, os.listdir(image_frames_folder)[0])  # Use the first frame as the image
    else:
        image_path = correct_audio_sample.replace('.wav', '.jpg')  # Use the corresponding image  

    question = {
        "question": f"Which sound is most likely made by the {object_name} from this image? Answer the question with A, B, C, or D",
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
    
    return question

def generate_question_vision_text(audio_choices, audio_paths, correct_answer, frame_in_folder=True):
    
    if frame_in_folder:
        image_frames_folders = [audio_sample.replace('.wav', '_frames') for audio_sample in audio_paths]
        image_paths = [os.path.join(folder, os.listdir(folder)[0]) for folder in image_frames_folders]
    else:
        image_paths = [audio_sample.replace('.wav', '.jpg') for audio_sample in audio_paths]    
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
    
    return question

def sample_instances(audio_choices, root_dir, mode='classes'):
    """
    Sample audio instances from the given choices.
    If mode is 'classes', it samples one instance per class.
    If mode is 'instances', it samples the exact instances provided.
    """
    all_audio_choices = []
    
    if mode == 'classes':
        for audio_choice in audio_choices:
            class_dir = os.path.join(root_dir, audio_choice)
            if os.path.isdir(class_dir):
                audio_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                if audio_files:
                    sampled_file = random.choice(audio_files)
                    all_audio_choices.append(os.path.join(class_dir, sampled_file))
                else:
                    print(f"No audio files found in {class_dir}")
    
    elif mode == 'instances':
        all_audio_choices = [os.path.join(root_dir, audio_choice) for audio_choice in audio_choices if audio_choice.endswith('.wav')]
    
    return all_audio_choices
if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    
    DATASET_NAME = 'landscapes'  # or 'vggss'
    OBJECT_NAME = 'objects'
    OBJECT_NAME = 'natural scenes'
    # DATASET_NAME = 'URMP'  # or 'vggss'
    root_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/landscape_audiobench/test_processed"
    mode = 'classes' # or instances
    N = 500
    
    
    export_dir = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/{DATASET_NAME}'
    if mode == 'classes':
        all_choices = os.listdir(root_dir)
    elif mode == 'instances':
        all_choices = os.listdir(root_dir)
        all_choices = [audio_name for audio_name in all_choices if audio_name.endswith('.wav')]   

    print(f"Total valid audio choices: {len(all_choices)}")
    
    
    audio_questions = []
    vision_questions = []
    for i in range(N):
        audio_choices = random.sample(all_choices, 4)
        
        audio_paths = sample_instances(audio_choices, root_dir, mode=mode)

        correct_answers = random.choice(['A', 'B', 'C', 'D'])

        question_vision = \
            generate_question_vision_audio(audio_choices, audio_paths, correct_answers, frame_in_folder=False, object_name=OBJECT_NAME)
        
        question_audio = \
            generate_question_audio_vision(audio_choices, audio_paths, correct_answers, frame_in_folder=False,  object_name=OBJECT_NAME)

        
        audio_questions.append(question_audio)
        vision_questions.append(question_vision)

    os.makedirs(export_dir, exist_ok=True)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_audio_vision.json", "w") as f:
        json.dump(audio_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_vision_audio.json", "w") as f:
        json.dump(vision_questions, f, indent=4)
    