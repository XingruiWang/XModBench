import os
import re
import random
import json


# ==================== 6 Question Types: Audio, Vision, Text Cross-Modal ====================

def generate_question_audio_vision(audio_choices, audio_paths, correct_answer, frame_in_folder=True, object_name='objects'):
    """Generate audio -> vision question: listen to sound and choose the matching image."""
    
    if frame_in_folder:
        image_frames_folders = [audio_sample.replace('.wav', '_frames') for audio_sample in audio_paths]
        image_paths = [os.path.join(folder, os.listdir(folder)[0]) for folder in image_frames_folders]
    else:
        image_paths = [audio_sample.replace('.wav', '.jpg') for audio_sample in audio_paths]    
    
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    correct_idx = ord(correct_answer) - ord('A')
    
    question = {
        "question": f"Listen to this audio. Which image most likely shows the {object_name} that makes this sound? Answer with A, B, C, or D",
        "conditions": {
            "modality": "Audio",
            "input": correct_audio_sample,
            "class": audio_choices[correct_idx]
        },
        "options": {
            "A": {"modality": "Image", "input": image_paths[0], "class": audio_choices[0]},
            "B": {"modality": "Image", "input": image_paths[1], "class": audio_choices[1]},
            "C": {"modality": "Image", "input": image_paths[2], "class": audio_choices[2]},
            "D": {"modality": "Image", "input": image_paths[3], "class": audio_choices[3]}
        },
        "correct_answer": correct_answer,
        "correct_class": audio_choices[correct_idx]
    }
    return question


def generate_question_vision_audio(audio_choices, audio_paths, correct_answer, frame_in_folder=True, object_name='objects'):
    """Generate vision -> audio question: look at image and choose the matching sound."""
    
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    correct_idx = ord(correct_answer) - ord('A')
    
    if frame_in_folder:
        image_frames_folder = correct_audio_sample.replace('.wav', '_frames')
        image_path = os.path.join(image_frames_folder, os.listdir(image_frames_folder)[0])
    else:
        image_path = correct_audio_sample.replace('.wav', '.jpg')

    question = {
        "question": f"Look at this image. Which audio clip is most likely produced by the {object_name} you see? Answer with A, B, C, or D",
        "conditions": {
            "modality": "Image",
            "input": image_path,
            "class": audio_choices[correct_idx]
        },
        "options": {
            "A": {"modality": "Audio", "input": audio_paths[0], "class": audio_choices[0]},
            "B": {"modality": "Audio", "input": audio_paths[1], "class": audio_choices[1]},
            "C": {"modality": "Audio", "input": audio_paths[2], "class": audio_choices[2]},
            "D": {"modality": "Audio", "input": audio_paths[3], "class": audio_choices[3]}
        },
        "correct_answer": correct_answer,
        "correct_class": audio_choices[correct_idx]
    }
    return question


def generate_question_audio_text(audio_choices, audio_paths, correct_answer, object_name='objects'):
    """Generate audio -> text question: listen to sound and choose the matching description."""
    
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    correct_idx = ord(correct_answer) - ord('A')
    
    question = {
        "question": f"Listen to this audio. Which text description best matches the {object_name} that makes this sound? Answer with A, B, C, or D",
        "conditions": {
            "modality": "Audio",
            "input": correct_audio_sample,
            "class": audio_choices[correct_idx]
        },
        "options": {
            "A": {"modality": "Text", "input": audio_choices[0]},
            "B": {"modality": "Text", "input": audio_choices[1]},
            "C": {"modality": "Text", "input": audio_choices[2]},
            "D": {"modality": "Text", "input": audio_choices[3]}
        },
        "correct_answer": correct_answer,
        "correct_class": audio_choices[correct_idx]
    }
    return question


def generate_question_text_audio(audio_choices, audio_paths, correct_answer, object_name='objects'):
    """Generate text -> audio question: read description and choose the matching sound."""
    
    correct_idx = ord(correct_answer) - ord('A')
    correct_class = audio_choices[correct_idx]
    
    question = {
        "question": f"Based on this description: '{correct_class}', which audio clip best matches the {object_name} described? Answer with A, B, C, or D",
        "conditions": {
            "modality": "Text",
            "input": correct_class,
        },
        "options": {
            "A": {"modality": "Audio", "input": audio_paths[0], "class": audio_choices[0]},
            "B": {"modality": "Audio", "input": audio_paths[1], "class": audio_choices[1]},
            "C": {"modality": "Audio", "input": audio_paths[2], "class": audio_choices[2]},
            "D": {"modality": "Audio", "input": audio_paths[3], "class": audio_choices[3]}
        },
        "correct_answer": correct_answer,
        "correct_class": correct_class
    }
    return question


def generate_question_text_vision(audio_choices, audio_paths, correct_answer, frame_in_folder=True, object_name='objects'):
    """Generate text -> vision question: read description and choose the matching image."""
    
    if frame_in_folder:
        image_frames_folders = [audio_sample.replace('.wav', '_frames') for audio_sample in audio_paths]
        image_paths = [os.path.join(folder, os.listdir(folder)[0]) for folder in image_frames_folders]
    else:
        image_paths = [audio_sample.replace('.wav', '.jpg') for audio_sample in audio_paths]
    
    correct_idx = ord(correct_answer) - ord('A')
    correct_class = audio_choices[correct_idx]
    
    question = {
        "question": f"Based on this description: '{correct_class}', which image best matches the {object_name} described? Answer with A, B, C, or D",
        "conditions": {
            "modality": "Text",
            "input": correct_class,
        },
        "options": {
            "A": {"modality": "Image", "input": image_paths[0], "class": audio_choices[0]},
            "B": {"modality": "Image", "input": image_paths[1], "class": audio_choices[1]},
            "C": {"modality": "Image", "input": image_paths[2], "class": audio_choices[2]},
            "D": {"modality": "Image", "input": image_paths[3], "class": audio_choices[3]}
        },
        "correct_answer": correct_answer,
        "correct_class": correct_class
    }
    return question


def generate_question_vision_text(audio_choices, audio_paths, correct_answer, frame_in_folder=True, object_name='objects'):
    """Generate vision -> text question: look at image and choose the matching description."""
    
    correct_idx = ord(correct_answer) - ord('A')
    correct_audio_sample = audio_paths[correct_idx]
    
    if frame_in_folder:
        image_frames_folder = correct_audio_sample.replace('.wav', '_frames')
        image_path = os.path.join(image_frames_folder, os.listdir(image_frames_folder)[0])
    else:
        image_path = correct_audio_sample.replace('.wav', '.jpg')
    
    question = {
        "question": f"Look at this image. Which text description best matches the {object_name} you see? Answer with A, B, C, or D",
        "conditions": {
            "modality": "Image",
            "input": image_path,
            "class": audio_choices[correct_idx]
        },
        "options": {
            "A": {"modality": "Text", "input": audio_choices[0]},
            "B": {"modality": "Text", "input": audio_choices[1]},
            "C": {"modality": "Text", "input": audio_choices[2]},
            "D": {"modality": "Text", "input": audio_choices[3]}
        },
        "correct_answer": correct_answer,
        "correct_class": audio_choices[correct_idx]
    }
    return question


# ==================== Helper Functions ====================

def check_valid_close_instruments(audio_choices):
    """Ensure no more than one instrument from similar/close instrument groups per question."""
    # Define groups of similar/close instruments
    close_instrument_groups = [
        ['Violin', 'Viola'],  # String instruments (similar)
        ['Flute', 'Oboe', 'Clarinet', 'Bassoon'],  # Woodwinds
        ['Trumpet', 'Trombone'],
        ['Horn', 'Tuba'],  # Brass instruments
        ['Cello', 'DoubleBass']  # Lower strings
    ]
    
    # Check each group for multiple instruments
    for group in close_instrument_groups:
        count_in_group = 0
        for audio_choice in audio_choices:
            if audio_choice in group:
                count_in_group += 1
        if count_in_group > 1:
            return False
    
    return True

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


# ==================== Main Generation Function ====================

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    
    DATASET_NAME = 'landscapes'
    OBJECT_NAME = 'natural scenes'
    root_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/landscape_audiobench/test_processed"
    mode = 'classes'  # or instances
    N = 500
    
    export_dir = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/natures'
    
    if mode == 'classes':
        all_choices = os.listdir(root_dir)
    elif mode == 'instances':
        all_choices = os.listdir(root_dir)
        all_choices = [audio_name for audio_name in all_choices if audio_name.endswith('.wav')]   

    all_choices = [choice for choice in all_choices if choice not in ['wind_noise']]
    print(f"Total valid audio choices: {len(all_choices)}")
    
    # Initialize question lists for all 6 types
    audio_vision_questions = []
    vision_audio_questions = []
    audio_text_questions = []
    text_audio_questions = []
    text_vision_questions = []
    vision_text_questions = []
    
    for i in range(N):
        audio_choices = random.sample(all_choices, 4)
        while not check_valid_close_instruments(audio_choices):
            audio_choices = random.sample(all_choices, 4)
        
        audio_paths = sample_instances(audio_choices, root_dir, mode=mode)
        correct_answer = random.choice(['A', 'B', 'C', 'D'])
        
        # Generate all 6 question types
        q1 = generate_question_audio_vision(audio_choices, audio_paths, correct_answer, frame_in_folder=False, object_name=OBJECT_NAME)
        q2 = generate_question_vision_audio(audio_choices, audio_paths, correct_answer, frame_in_folder=False, object_name=OBJECT_NAME)
        q3 = generate_question_audio_text(audio_choices, audio_paths, correct_answer, object_name=OBJECT_NAME)
        q4 = generate_question_text_audio(audio_choices, audio_paths, correct_answer, object_name=OBJECT_NAME)
        q5 = generate_question_text_vision(audio_choices, audio_paths, correct_answer, frame_in_folder=False, object_name=OBJECT_NAME)
        q6 = generate_question_vision_text(audio_choices, audio_paths, correct_answer, frame_in_folder=False, object_name=OBJECT_NAME)
        
        audio_vision_questions.append(q1)
        vision_audio_questions.append(q2)
        audio_text_questions.append(q3)
        text_audio_questions.append(q4)
        text_vision_questions.append(q5)
        vision_text_questions.append(q6)

    # Save all question types
    os.makedirs(export_dir, exist_ok=True)
    
    with open(f"{export_dir}/{DATASET_NAME}_audio_to_vision.json", "w") as f:
        json.dump(audio_vision_questions, f, indent=4)
    
    with open(f"{export_dir}/{DATASET_NAME}_vision_to_audio.json", "w") as f:
        json.dump(vision_audio_questions, f, indent=4)
    
    with open(f"{export_dir}/{DATASET_NAME}_audio_to_text.json", "w") as f:
        json.dump(audio_text_questions, f, indent=4)
    
    with open(f"{export_dir}/{DATASET_NAME}_text_to_audio.json", "w") as f:
        json.dump(text_audio_questions, f, indent=4)
    
    with open(f"{export_dir}/{DATASET_NAME}_text_to_vision.json", "w") as f:
        json.dump(text_vision_questions, f, indent=4)
    
    with open(f"{export_dir}/{DATASET_NAME}_vision_to_text.json", "w") as f:
        json.dump(vision_text_questions, f, indent=4)
    
    print(f"Generated {N} questions for each of the 6 question types:")
    print("1. Audio → Vision")
    print("2. Vision → Audio") 
    print("3. Audio → Text")
    print("4. Text → Audio")
    print("5. Text → Vision")
    print("6. Vision → Text")
    print(f"\nFiles saved to: {export_dir}")