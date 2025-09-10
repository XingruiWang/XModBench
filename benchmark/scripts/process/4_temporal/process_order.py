import os
import re
import random
import json


# ==================== 1. Main function: Load model and answer one question ====================

def load_description(description_path, return_count = False):
    with open(description_path, 'r') as f:
        description = f.read().strip().split(',')
    description = [f"{i+1}. {line}" for i, line in enumerate(description)]
    count = len(description)
    description = '; '.join(description)
    if return_count:
        return description, count
    else:
        return description

def generate_question_audio_video(all_choices, correct_answer, object_name='objects'):

    video_paths = [os.path.join(choice, 'concat.mp4') for choice in all_choices]
    audio_paths = [os.path.join(choice, 'mixed.wav') for choice in all_choices]
    description_paths = [os.path.join(choice, 'order.txt') for choice in all_choices]
    descriptions = [load_description(description_path) for description_path in description_paths]
    
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    correct_description, count = load_description(description_paths[ord(correct_answer) - ord('A')], return_count=True)

    question = {
        "question": (
                        "Listen to this audio sequence where multiple sounds occur one after another in a specific temporal order. "
                        "Which video best represents the correct chronological sequence of events as they appear in the audio? "
                        "The video shows the temporal order of events occurring sequentially over time. "
                        "Answer the question with A, B, C, or D."
                    ),
        "conditions": {
            "modality": "Audio",
            "input": correct_audio_sample,
            "description": correct_description,
            },
        "options": {
            "A": {
                "modality": "Video",
                "input": video_paths[0],
                "description": descriptions[0],
            },
            "B": {
                "modality": "Video",
                "input": video_paths[1],
                "description": descriptions[1],
            },
            "C": {
                "modality": "Video",
                "input": video_paths[2],
                "description": descriptions[2],
            },
            "D": {
                "modality": "Video",
                "input": video_paths[3],
                "description": descriptions[3],
            }
        },
        "correct_answer": correct_answer,
        "extra_info": {
            "count": count
        }
    }
    
    return question


def generate_question_video_audio(all_choices, correct_answer, object_name='objects'):
    
    video_paths = [os.path.join(choice, 'concat.mp4') for choice in all_choices]
    audio_paths = [os.path.join(choice, 'mixed.wav') for choice in all_choices]
    description_paths = [os.path.join(choice, 'order.txt') for choice in all_choices]
    descriptions = [load_description(description_path) for description_path in description_paths]
    
    correct_video_path = video_paths[ord(correct_answer) - ord('A')]
    correct_description, count = load_description(description_paths[ord(correct_answer) - ord('A')], return_count=True)
    
    question = {
       "question": (
                        "The video below shows a sequence of events occurring in chronological order over time, "
                        "representing the temporal order in which they should occur. "
                        "Listen carefully to each audio clip and determine the temporal sequence of events. "
                        "Which audio clip matches the chronological order shown in the video? "
                        "Choose A, B, C, or D."
                    ),
        "conditions": {
            "modality": "Video",
            "input": correct_video_path,
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
    
def generate_question_video_text(all_choices, correct_answer, object_name='objects'):
    """
    Generate a question from video to text.
    Given a video, ask which text description most likely corresponds to their temporal sequence.
    """
    video_paths = [os.path.join(choice, 'concat.mp4') for choice in all_choices]
    description_paths = [os.path.join(choice, 'order.txt') for choice in all_choices]
    descriptions = [load_description(description_path) for description_path in description_paths]
    
    correct_video_path = video_paths[ord(correct_answer) - ord('A')]
    correct_description, count = load_description(description_paths[ord(correct_answer) - ord('A')], return_count=True)
    
    question = {
        "question": (
            "The video below shows a sequence of events occurring chronologically over time, "
            "representing the temporal order in which they occur. "
            "Which text description best corresponds to this temporal sequence? "
            "Choose A, B, C, or D."
        ),
        "conditions": {
            "modality": "Video",
            "input": correct_video_path,
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

def generate_question_text_video(all_choices, correct_answer, object_name='objects'):
    """
    Generate a question from text to video.
    Given a text description of the temporal sequence of instruments, ask which video most likely corresponds to the chronological order.
    """
    video_paths = [os.path.join(choice, 'concat.mp4') for choice in all_choices]
    description_paths = [os.path.join(choice, 'order.txt') for choice in all_choices]
    descriptions = [load_description(description_path) for description_path in description_paths]
    
    correct_text, count = load_description(description_paths[ord(correct_answer) - ord('A')], return_count=True)
    correct_video_path = video_paths[ord(correct_answer) - ord('A')]
    
    question = {
        "question": (
            "The text below describes the chronological sequence of events occurring over time (from first to last). "
            "Which video best represents this temporal order, with events occurring sequentially over time? "
            "Answer the question with A, B, C, or D."
        ),
        "conditions": {
            "modality": "Text",
            "input": correct_text,
            "description": correct_text,
        },
        "options": {
            "A": {
                "modality": "Video",
                "input": video_paths[0],
                "description": descriptions[0],
            },
            "B": {
                "modality": "Video",
                "input": video_paths[1],
                "description": descriptions[1],
            },
            "C": {
                "modality": "Video",
                "input": video_paths[2],
                "description": descriptions[2],
            },
            "D": {
                "modality": "Video",
                "input": video_paths[3],
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
    Given an audio clip, ask which text description most likely corresponds to the temporal sequence.
    """
    audio_paths = [os.path.join(choice, 'mixed.wav') for choice in all_choices]
    description_paths = [os.path.join(choice, 'order.txt') for choice in all_choices]
    descriptions = [load_description(description_path) for description_path in description_paths]
    
    correct_audio_path = audio_paths[ord(correct_answer) - ord('A')]
    correct_description, count = load_description(description_paths[ord(correct_answer) - ord('A')], return_count=True)
    
    question = {
        "question": (
            "Listen to the audio clip below, which features multiple sounds occurring in a specific temporal sequence. "
            "Pay close attention to the chronological order in which each sound appears. "
            "Which text best describes the temporal order of the events (from first to last)? "
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
    Given a text description, ask which audio clip most likely corresponds to the temporal sequence.
    """
    audio_paths = [os.path.join(choice, 'mixed.wav') for choice in all_choices]
    description_paths = [os.path.join(choice, 'order.txt') for choice in all_choices]
    descriptions = [load_description(description_path) for description_path in description_paths]
    
    correct_text, count = load_description(description_paths[ord(correct_answer) - ord('A')], return_count=True)
    correct_audio_path = audio_paths[ord(correct_answer) - ord('A')]
    question = {
        "question": (
            "The text below describes the temporal sequence in which several events occur over time (from first to last). "
            "Listen carefully to each audio clip and pay attention to the chronological order of the sounds. "
            "Which audio clip best matches this temporal sequence? "
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
    
    DATASET_NAME = 'vggss_order'  # or 'vggss'
    OBJECT_NAME = 'objects'

    root_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/temporal_audiobench"
    
    N = 0
    
    export_dir = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/04_temporal/order'

    audio_video_questions = []
    video_audio_questions = []
    video_text_questions = []
    text_video_questions = []
    text_audio_questions = []
    audio_text_questions = []
     
     
    all_backup_choices = []
    for instance in os.listdir(root_dir):
        if not instance.startswith('sample'):
            continue
        all_choices = [os.path.join(root_dir, instance, f) for f in os.listdir(os.path.join(root_dir, instance))]
        if len(all_choices) != 4:
            print(f"Warning: {instance} has {len(all_choices)} choices")
        all_backup_choices.extend(all_choices)
        
    print(f"Found {len(all_backup_choices)} backup choices")
    
    all_backup_choices = set(all_backup_choices)
    
    for instance in os.listdir(root_dir):
        if not instance.startswith('sample'):
            continue
        all_choices = [os.path.join(root_dir, instance, f) for f in os.listdir(os.path.join(root_dir, instance))]
        
        i = 0
        choice = all_choices[i]
        
        other_choices = all_choices[:i] + all_choices[i+1:]

        choises = [choice] + other_choices     
    
        # 
        order = [0, 1, 2, 3]
        random.shuffle(order)
        
        correct_answer = ['A', 'B', 'C', 'D'][order.index(0)]
        
        choises = [choises[i] for i in order]
        
        question_audio_video = generate_question_audio_video(choises, correct_answer)
        question_video_audio = generate_question_video_audio(choises, correct_answer)
        question_video_text = generate_question_video_text(choises, correct_answer)
        question_text_video = generate_question_text_video(choises, correct_answer)
        question_audio_text = generate_question_audio_text(choises, correct_answer)
        question_text_audio = generate_question_text_audio(choises, correct_answer)
        
        audio_video_questions.append(question_audio_video)
        video_audio_questions.append(question_video_audio)
        video_text_questions.append(question_video_text)
        text_video_questions.append(question_text_video)
        text_audio_questions.append(question_text_audio)
        audio_text_questions.append(question_audio_text)
        
        N += 1
        if N >= 500:
            break
    os.makedirs(export_dir, exist_ok=True)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_audio_vision.json", "w") as f:
        json.dump(audio_video_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_vision_audio.json", "w") as f:
        json.dump(video_audio_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_vision_text.json", "w") as f:
        json.dump(video_text_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_text_vision.json", "w") as f:
        json.dump(text_video_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_audio_text.json", "w") as f:
        json.dump(text_audio_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_text_audio.json", "w") as f:
        json.dump(audio_text_questions, f, indent=4)
    print(f"Generated {N} questions")
    print(f"Saved to {export_dir}")