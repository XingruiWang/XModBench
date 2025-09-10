import os
import re
import random
import json


# ==================== 1. Main function: Load model and answer one question ====================

def load_description(description_path, return_count = False):
    with open(description_path, 'r') as f:
        description = f.readlines()
        description = [line.strip() for line in description if line.strip()]
    count = len(description)
    description = 'From left to right: ' + ', '.join(description)
    if return_count:
        return description, count
    else:
        return description

def generate_question_audio_vision(all_choices, correct_answer, object_name='objects'):

    image_paths = [os.path.join(choice, 'image.jpg') for choice in all_choices]
    audio_paths = [os.path.join(choice, 'audio.wav') for choice in all_choices]
    description_paths = [os.path.join(choice, 'objects.txt') for choice in all_choices]
    descriptions = [load_description(description_path) for description_path in description_paths]
    
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    correct_description, count = load_description(description_paths[ord(correct_answer) - ord('A')], return_count=True)

    question = {
        "question": (
                        "This audio is a stereo recording of several instruments playing from left to right. "
                        "Which image best represents the correct spatial layout of the scene based on the sound? "
                        "Answer the question with A, B, C, or D."
                    ),
        "conditions": {
            "modality": "Audio",
            "input": correct_audio_sample,
            "description": correct_description,
            },
        "options": {
            "A": {
                "modality": "Image",
                "input": image_paths[0],
                "description": descriptions[0],
            },
            "B": {
                "modality": "Image",
                "input": image_paths[1],
                "description": descriptions[1],
            },
            "C": {
                "modality": "Image",
                "input": image_paths[2],
                "description": descriptions[2],
            },
            "D": {
                "modality": "Image",
                "input": image_paths[3],
                "description": descriptions[3],
            }
        },
        "correct_answer": correct_answer,
        "extra_info": {
            "count": count
        }
    }
    
    return question


def generate_question_vision_audio(all_choices, correct_answer, object_name='objects'):
    
    image_paths = [os.path.join(choice, 'image.jpg') for choice in all_choices]
    audio_paths = [os.path.join(choice, 'audio.wav') for choice in all_choices]
    description_paths = [os.path.join(choice, 'objects.txt') for choice in all_choices]
    descriptions = [load_description(description_path) for description_path in description_paths]
    
    correct_image_path = image_paths[ord(correct_answer) - ord('A')]
    correct_description, count = load_description(description_paths[ord(correct_answer) - ord('A')], return_count=True)
    
    question = {
       "question": (
                        "The image below shows several instruments arranged from left to right. "
                        "Listen carefully to each audio clip—pay attention to how the sounds are distributed between the left and right channels. "
                        "Which audio clip best matches the spatial layout shown in the image? "
                        "Choose A, B, C, or D."
                    ),
        "conditions": {
            "modality": "Image",
            "input": correct_image_path,
            "description": correct_description,
        },
        "options": {
            "A": {
                "modality": "Audio",
                "input": audio_paths[0],
                "description": descriptions[0],
            },
            "B": {
                "modality": "Audio",
                "input": audio_paths[1],
                "description": descriptions[1],
            },
            "C": {
                "modality": "Audio",
                "input": audio_paths[2],
                "description": descriptions[2],
            },
            "D": {
                "modality": "Audio",
                "input": audio_paths[3],    
                "description": descriptions[3],
            }
        },
        "correct_answer": correct_answer,
        "extra_info": {
            "count": count
        }
    }
    
    return question 
    
def generate_question_vision_text(all_choices, correct_answer, object_name='objects'):
    """
    Generate a question from vision to text.
    Given an image, ask which text description most likely corresponds to their spatial arrangement from left to right.
    """
    image_paths = [os.path.join(choice, 'image.jpg') for choice in all_choices]
    description_paths = [os.path.join(choice, 'objects.txt') for choice in all_choices]
    descriptions = [load_description(description_path) for description_path in description_paths]
    
    correct_image_path = image_paths[ord(correct_answer) - ord('A')]
    correct_description, count = load_description(description_paths[ord(correct_answer) - ord('A')], return_count=True)
    
    question = {
        "question": (
            "The image below shows several instruments arranged from left to right. "
            "Which text description best corresponds to this spatial arrangement? "
            "Choose A, B, C, or D."
        ),
        "conditions": {
            "modality": "Image",
            "input": correct_image_path,
            "description": correct_description
        },
        "options": {
            "A": {
                "modality": "Text",
                "input": descriptions[0],
                "description": descriptions[0],
            },
            "B": {
                "modality": "Text",
                "input": descriptions[1],
                "description": descriptions[1],
            },
            "C": {
                "modality": "Text",
                "input": descriptions[2],
                "description": descriptions[2],
            },
            "D": {
                "modality": "Text",
                "input": descriptions[3],
                "description": descriptions[3],
            }
        },
        "correct_answer": correct_answer,
        "extra_info": {
            "count": count
        }
    }
    
    return question

def generate_question_text_vision(all_choices, correct_answer, object_name='objects'):
    """
    Generate a question from text to vision.
    Given a text description of the spatial arrangement of instruments from left to right, ask which image most likely corresponds to the spatial arrangement.
    """
    image_paths = [os.path.join(choice, 'image.jpg') for choice in all_choices]
    description_paths = [os.path.join(choice, 'objects.txt') for choice in all_choices]
    descriptions = [load_description(description_path) for description_path in description_paths]
    
    correct_text, count = load_description(description_paths[ord(correct_answer) - ord('A')], return_count=True)
    correct_image_path = image_paths[ord(correct_answer) - ord('A')]
    
    question = {
        "question": (
            "The text below describes the spatial arrangement of instruments from left to right. "
            "Which image best matches this left-to-right layout? "
            "Answer the question with A, B, C, or D."
        ),
        "conditions": {
            "modality": "Text",
            "input": correct_text,
            "description": correct_text,
        },
        "options": {
            "A": {
                "modality": "Image",
                "input": image_paths[0],
                "description": descriptions[0],
            },
            "B": {
                "modality": "Image",
                "input": image_paths[1],
                "description": descriptions[1],
            },
            "C": {
                "modality": "Image",
                "input": image_paths[2],
                "description": descriptions[2],
            },
            "D": {
                "modality": "Image",
                "input": image_paths[3],
                "description": descriptions[3],
            }
        },
        "correct_answer": correct_answer,
        "extra_info": {
            "count": count
        }
    }
    
    return question

def generate_question_audio_text(all_choices, correct_answer, object_name='objects'):
    """
    Generate a question from audio to text.
    Given an audio clip, ask which text description most likely corresponds to the spatial arrangement.
    """
    audio_paths = [os.path.join(choice, 'audio.wav') for choice in all_choices]
    description_paths = [os.path.join(choice, 'objects.txt') for choice in all_choices]
    descriptions = [load_description(description_path) for description_path in description_paths]
    
    correct_audio_path = audio_paths[ord(correct_answer) - ord('A')]
    correct_description, count = load_description(description_paths[ord(correct_answer) - ord('A')], return_count=True)
    
    question = {
        "question": (
            "The audio clip below features multiple instruments arranged from left to right. "
            "Pay close attention to how each sound appears in the stereo field. "
            "Which text best describes the spatial pattern? "
            "Choose A, B, C, or D."
        ),
        "conditions": {
            "modality": "Audio",
            "input": correct_audio_path,
            "description": correct_description,
        },
        "options": {
            "A": {
                "modality": "Text",
                "input": descriptions[0],
                "description": descriptions[0],
            },
            "B": {
                "modality": "Text",
                "input": descriptions[1],
                "description": descriptions[1],
            },
            "C": {
                "modality": "Text",
                "input": descriptions[2],
                "description": descriptions[2],
            },
            "D": {
                "modality": "Text",
                "input": descriptions[3],
                "description": descriptions[3],
            }
        },
        "correct_answer": correct_answer,
        "extra_info": {
            "count": count
        }
    }
    
    return question

def generate_question_text_audio(all_choices, correct_answer, object_name='objects'):
    """
    Generate a question from text to audio.
    Given a text description, ask which audio clip most likely corresponds to the spatial arrangement.
    """
    audio_paths = [os.path.join(choice, 'audio.wav') for choice in all_choices]
    description_paths = [os.path.join(choice, 'objects.txt') for choice in all_choices]
    descriptions = [load_description(description_path) for description_path in description_paths]
    
    correct_text, count = load_description(description_paths[ord(correct_answer) - ord('A')], return_count=True)
    correct_audio_path = audio_paths[ord(correct_answer) - ord('A')]
    
    question = {
        "question": (
            "The text below describes how several instruments are arranged from left to right. "
            "Listen carefully to each audio clip—focus on how the sounds are distributed across the left and right channels. "
            "Which audio clip best matches this spatial arrangement? "
            "Choose A, B, C, or D."
        ),
        "conditions": {
            "modality": "Text",
            "input": correct_text,
            "description": correct_text,
        },
        "options": {
            "A": {
                "modality": "Audio",
                "input": audio_paths[0],
                "description": descriptions[0],
            },
            "B": {
                "modality": "Audio",
                "input": audio_paths[1],
                "description": descriptions[1],
            },
            "C": {
                "modality": "Audio",
                "input": audio_paths[2],
                "description": descriptions[2],
            },
            "D": {
                "modality": "Audio",
                "input": audio_paths[3],
                "description": descriptions[3],
            }
        },
        "correct_answer": correct_answer,
        "extra_info": {
            "count": count
        }
    }
    
    return question



if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    
    DATASET_NAME = 'urmp'  # or 'vggss'
    OBJECT_NAME = 'objects'

    root_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/URMP_processed_easy"
    
    N = 0
    
    export_dir = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/02_spatial/{DATASET_NAME}_easy'

    audio_vision_questions = []
    vision_audio_questions = []
    vision_text_questions = []
    text_vision_questions = []
    text_audio_questions = []
    audio_text_questions = []
     
     
    all_backup_choices = []
    for instance in os.listdir(root_dir):
        all_choices = [os.path.join(root_dir, instance, 'reordered', f) for f in os.listdir(os.path.join(root_dir, instance, 'reordered'))]
        all_backup_choices.extend(all_choices)
    all_backup_choices = set(all_backup_choices)
    
    for instance in os.listdir(root_dir):
        all_choices = [os.path.join(root_dir, instance, 'reordered', f) for f in os.listdir(os.path.join(root_dir, instance, 'reordered'))]
        
        for i in range(len(all_choices)):
            choice = all_choices[i]
            
            other_choices = all_choices[:i] + all_choices[i+1:]
            
            if len(other_choices) <= 3:
                extra_N = 4 - len(other_choices)
                extra_choices  = all_backup_choices - set(all_choices)
                
                extra_choices = random.sample(list(extra_choices), extra_N)
                other_choices.extend(extra_choices)
                
            choises = [choice] + other_choices     
        
            # 
            order = [0, 1, 2, 3]
            order = random.sample(order, 4)
            
            correct_answer = ['A', 'B', 'C', 'D'][order.index(0)]
            
            choises = [choises[i] for i in order]
            
            question_audio_vision = generate_question_audio_vision(choises, correct_answer)
            question_vision_audio = generate_question_vision_audio(choises, correct_answer)
            question_vision_text = generate_question_vision_text(choises, correct_answer)
            question_text_vision = generate_question_text_vision(choises, correct_answer)
            question_audio_text = generate_question_audio_text(choises, correct_answer)
            question_text_audio = generate_question_text_audio(choises, correct_answer)
            
            audio_vision_questions.append(question_audio_vision)
            vision_audio_questions.append(question_vision_audio)
            vision_text_questions.append(question_vision_text)
            text_vision_questions.append(question_text_vision)
            text_audio_questions.append(question_text_audio)
            audio_text_questions.append(question_audio_text)
            
            N += 1
            print(f"Generated {N} questions")
            if N >= 500:
                break
                
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
                  
                
                
                
        
        
        
        