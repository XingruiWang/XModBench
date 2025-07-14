import os
import re
import random
import json


# ==================== 1. Main function: Load model and answer one question ====================


def generate_question_audio_vision(audio_paths, correct_answer, frame_in_folder=True, object_name='objects'):
    
    if frame_in_folder:
        image_frames_folders = [audio_sample.replace('.wav', '_frames') for audio_sample in audio_paths]
        image_paths = [os.path.join(folder, os.listdir(folder)[0]) for folder in image_frames_folders]
    else:
        image_paths = [audio_sample.replace('.wav', '.png') for audio_sample in audio_paths] 
           
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    
    question = {
        "question": f"Which image most likely written the same text as spoken in this audio? Answer the question with A, B, C, or D",
        "conditions": {
            "modality": "Audio",
            "input": correct_audio_sample,
            "text": read_text(correct_audio_sample.replace('.wav', '.txt'))
            },
        "options": {
            "A": {
                "modality": "Image",
                "input": image_paths[0],
                "text": read_text(image_paths[0].replace('.png', '.txt'))
            },
            "B": {
                "modality": "Image",
                "input": image_paths[1],
                "text": read_text(image_paths[1].replace('.png', '.txt'))
            },
            "C": {
                "modality": "Image",
                "input": image_paths[2],
                "text": read_text(image_paths[2].replace('.png', '.txt'))
            },
            "D": {
                "modality": "Image",
                "input": image_paths[3],
                "text": read_text(image_paths[3].replace('.png', '.txt'))
            }
        },
        "correct_answer": correct_answer
    }
    
    return question


def generate_question_vision_audio( audio_paths, correct_answer, frame_in_folder=True, object_name='objects'):
    
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    
    # image_frames_folder = correct_audio_sample.replace('.wav', '_frames')
    # image_path = os.path.join(image_frames_folder, os.listdir(image_frames_folder)[0])  # Use the first frame as the image
    if frame_in_folder:
        image_frames_folder = correct_audio_sample.replace('.wav', '_frames')
        image_path = os.path.join(image_frames_folder, os.listdir(image_frames_folder)[0])  # Use the first frame as the image
    else:
        image_path = correct_audio_sample.replace('.wav', '.png') # Use the corresponding image  

    question = {
        "question": f"Which audio most likely spoken the same text as written in this image? Answer the question with A, B, C, or D",
        "conditions": {
            "modality": "Image",
            "input": image_path,
            "text": image_path.replace('.png', '.txt')
            },
        "options": {
            "A": {
                "modality": "Audio",
                "input": audio_paths[0],
                "text": read_text(audio_paths[0].replace('.wav', '.txt'))
            },
            "B": {
                "modality": "Audio",
                "input": audio_paths[1],
                "text": read_text(audio_paths[1].replace('.wav', '.txt'))
            },
            "C": {
                "modality": "Audio",
                "input": audio_paths[2],
                "text": read_text(audio_paths[2].replace('.wav', '.txt'))
            },
            "D": {
                "modality": "Audio",
                "input": audio_paths[3],
                "text": read_text(audio_paths[3].replace('.wav', '.txt'))
            }
        },
        "correct_answer": correct_answer
    }
    
    return question

def read_text(path):
    with open(path, 'r') as f:
        return f.read()
    
def generate_question_vision_text(choices_paths, correct_answer, frame_in_folder=True, object_name='objects'):
    

    
    if frame_in_folder:
        image_frames_folder = choices_paths[ord(correct_answer) - ord('A')].replace('.wav', '_frames')
        image_path = os.path.join(image_frames_folder, os.listdir(image_frames_folder)[0])  # Use the first frame as the image
    else:
        image_path = choices_paths[ord(correct_answer) - ord('A')].replace('.wav', '.png') # Use the corresponding image  

    question = {
        "question": f"Which text most likely written the same content as shown in this image? Answer the question with A, B, C, or D",
        "conditions": {
            "modality": "Image",
            "input": image_path,
            "text": read_text(image_path.replace('.png', '.txt'))
            },
        "options": {
            "A": {
                "modality": "Text",
                "input": read_text(choices_paths[0].replace('.wav', '.txt')),
                "text": read_text(choices_paths[0].replace('.wav', '.txt'))
            },
            "B": {
                "modality": "Text",
                "input": read_text(choices_paths[1].replace('.wav', '.txt')),
                "text": read_text(choices_paths[1].replace('.wav', '.txt'))
            },
            "C": {
                "modality": "Text",
                "input": read_text(choices_paths[2].replace('.wav', '.txt')),
                "text": read_text(choices_paths[2].replace('.wav', '.txt'))
            },
            "D": {
                "modality": "Text",
                "input": read_text(choices_paths[3].replace('.wav', '.txt')),
                "text": read_text(choices_paths[3].replace('.wav', '.txt'))
            }
        },
        "correct_answer": correct_answer
    }
    
    return question
    
    

def generate_question_text_vision(choices_paths, correct_answer, frame_in_folder=True, object_name='objects'):
    """
    Generate a question from text to image (vision).
    Given a text, ask which image most likely contains the same content.
    """
    
    correct_text_path = choices_paths[ord(correct_answer) - ord('A')].replace('.wav', '.txt')
    
    question = {
        "question": f"Which image most likely contains the same content as this text? Answer the question with A, B, C, or D",
        "conditions": {
            "modality": "Text",
            "input": read_text(correct_text_path),
            "text": read_text(correct_text_path)
        },
        "options": {
            "A": {
                "modality": "Image",
                "input": choices_paths[0].replace('.wav', '.png') if not frame_in_folder else os.path.join(choices_paths[0].replace('.wav', '_frames'), os.listdir(choices_paths[0].replace('.wav', '_frames'))[0]),
                "text": read_text(choices_paths[0].replace('.wav', '.txt'))
            },
            "B": {
                "modality": "Image", 
                "input": choices_paths[1].replace('.wav', '.png') if not frame_in_folder else os.path.join(choices_paths[1].replace('.wav', '_frames'), os.listdir(choices_paths[1].replace('.wav', '_frames'))[0]),
                "text": read_text(choices_paths[1].replace('.wav', '.txt'))
            },
            "C": {
                "modality": "Image",
                "input": choices_paths[2].replace('.wav', '.png') if not frame_in_folder else os.path.join(choices_paths[2].replace('.wav', '_frames'), os.listdir(choices_paths[2].replace('.wav', '_frames'))[0]),
                "text": read_text(choices_paths[2].replace('.wav', '.txt'))
            },
            "D": {
                "modality": "Image",
                "input": choices_paths[3].replace('.wav', '.png') if not frame_in_folder else os.path.join(choices_paths[3].replace('.wav', '_frames'), os.listdir(choices_paths[3].replace('.wav', '_frames'))[0]),
                "text": read_text(choices_paths[3].replace('.wav', '.txt'))
            }
        },
        "correct_answer": correct_answer
    }
    
    return question

def generate_question_audio_text(choices_paths, correct_answer, frame_in_folder=True, object_name='objects'):
    """
    Generate a question from audio to text.
    Given an audio, ask which text most likely contains the same content.
    """
    correct_audio_path = choices_paths[ord(correct_answer) - ord('A')]
    
    question = {
        "question": f"Which text most likely contains the same content as this audio? Answer the question with A, B, C, or D",
        "conditions": {
            "modality": "Audio",
            "input": correct_audio_path,
            "text": read_text(correct_audio_path.replace('.wav', '.txt'))
        },
        "options": {
            "A": {
                "modality": "Text",
                "input": read_text(choices_paths[0].replace('.wav', '.txt')),
                "text": read_text(choices_paths[0].replace('.wav', '.txt'))
            },
            "B": {
                "modality": "Text",
                "input": read_text(choices_paths[1].replace('.wav', '.txt')),
                "text": read_text(choices_paths[1].replace('.wav', '.txt'))
            },
            "C": {
                "modality": "Text",
                "input": read_text(choices_paths[2].replace('.wav', '.txt')),
                "text": read_text(choices_paths[2].replace('.wav', '.txt'))
            },
            "D": {
                "modality": "Text",
                "input": read_text(choices_paths[3].replace('.wav', '.txt')),
                "text": read_text(choices_paths[3].replace('.wav', '.txt'))
            }
        },
        "correct_answer": correct_answer
    }
    
    return question

def generate_question_text_audio(choices_paths, correct_answer, frame_in_folder=True, object_name='objects'):
    """
    Generate a question from text to audio.
    Given a text, ask which audio most likely contains the same content.
    """
    correct_text_path = choices_paths[ord(correct_answer) - ord('A')].replace('.wav', '.txt')
    
    question = {
        "question": f"Which audio most likely contains the same content as this text? Answer the question with A, B, C, or D",
        "conditions": {
            "modality": "Text",
            "input": read_text(correct_text_path),
            "text": read_text(correct_text_path)
        },
        "options": {
            "A": {
                "modality": "Audio",
                "input": choices_paths[0],
                "text": read_text(choices_paths[0].replace('.wav', '.txt'))
            },
            "B": {
                "modality": "Audio",
                "input": choices_paths[1],
                "text": read_text(choices_paths[1].replace('.wav', '.txt'))
            },
            "C": {
                "modality": "Audio",
                "input": choices_paths[2],
                "text": read_text(choices_paths[2].replace('.wav', '.txt'))
            },
            "D": {
                "modality": "Audio",
                "input": choices_paths[3],
                "text": read_text(choices_paths[3].replace('.wav', '.txt'))
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
    
    DATASET_NAME = 'acr'  # or 'vggss'
    OBJECT_NAME = 'text'

    # DATASET_NAME = 'URMP'  # or 'vggss'
    root_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/rendertext/audio"
    mode = 'instances' # or instances
    N = 1000
    
    
    # export_dir = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/{DATASET_NAME}'
    export_dir = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/03_ocr/{DATASET_NAME}'
    
    if mode == 'classes':
        all_choices = os.listdir(root_dir)
    
    elif mode == 'instances':
        all_choices = os.listdir(root_dir)
        all_choices = [audio_name for audio_name in all_choices if audio_name.endswith('.wav')]   

    print(f"Total valid audio choices: {len(all_choices)}")
    
    
    audio_vision_questions = []
    vision_audio_questions = []
    vision_text_questions = []
    text_vision_questions = []
    text_audio_questions = []
    audio_text_questions = []
    for i in range(N):
        audio_choices = random.sample(all_choices, 4)
        
        audio_paths = sample_instances(audio_choices, root_dir, mode=mode)

        correct_answers = random.choice(['A', 'B', 'C', 'D'])

        question_vision_audio = \
            generate_question_vision_audio(audio_paths, correct_answers, frame_in_folder=False, object_name=OBJECT_NAME)
        
        question_audio_vision = \
            generate_question_audio_vision( audio_paths, correct_answers, frame_in_folder=False,  object_name=OBJECT_NAME)
            
        question_vision_text = \
            generate_question_vision_text(audio_paths, correct_answers, frame_in_folder=False, object_name=OBJECT_NAME)
        
        question_text_vision = \
            generate_question_text_vision(audio_paths, correct_answers, frame_in_folder=False, object_name=OBJECT_NAME)
        
        question_text_audio = \
            generate_question_text_audio(audio_paths, correct_answers, frame_in_folder=False, object_name=OBJECT_NAME)
        
        question_audio_text = \
            generate_question_audio_text(audio_paths, correct_answers, frame_in_folder=False, object_name=OBJECT_NAME)
        
        audio_vision_questions.append(question_vision_audio)
        vision_audio_questions.append(question_audio_vision)
        vision_text_questions.append(question_vision_text)
        text_vision_questions.append(question_text_vision)
        text_audio_questions.append(question_text_audio)
        audio_text_questions.append(question_audio_text)

    os.makedirs(export_dir, exist_ok=True)

    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_audio_vision.json", "w") as f:
        json.dump(audio_vision_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_vision_audio.json", "w") as f:
        json.dump(vision_audio_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_vision_text.json", "w") as f:
        json.dump(vision_text_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_text_vision.json", "w") as f:
        json.dump(text_vision_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_text_audio.json", "w") as f:
        json.dump(text_audio_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_audio_text.json", "w") as f:
        json.dump(audio_text_questions, f, indent=4)
    