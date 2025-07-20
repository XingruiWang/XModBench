import os
import re
import random
import json


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

def generate_question_audio_text(text_choices, audio_paths, correct_answer):
    """
    Generate a question from audio to text.
    Given an audio clip, ask which text description most likely corresponds to the spatial arrangement.
    """

    correct_audio_path = audio_paths[ord(correct_answer) - ord('A')]
    
    question = {
        "question": (
            "Listen carefully to the audio clip and focus on how many repetitions of the action are in the audio. "
            "Which text best describes the number of repetitions? "
            "Choose A, B, C, or D."
        ),
        "conditions": {
            "modality": "Audio",
            "input": correct_audio_path,
        },
        "options": {
            "A": {
                "modality": "Text",
                "input": f"{text_choices[0]} times",
            },
            "B": {
                "modality": "Text",
                "input": f"{text_choices[1]} times",
            },
            "C": {
                "modality": "Text",
                "input": f"{text_choices[2]} times",
            },
            "D": {
                "modality": "Text",
                "input": f"{text_choices[3]} times",
            }
        },
        "correct_answer": correct_answer,
    }
    
    return question

def generate_question_text_audio(text_choices, audio_paths, correct_answer):
    """
    Generate a question from text to audio.
    Given a text description, ask which audio clip most likely corresponds to the spatial arrangement.
    """

    correct_text = text_choices[ord(correct_answer) - ord('A')]
    
    question = {
        "question": (
            "Listen carefully to each audio clip—focus on how many repetitions of the action are in the audio. "
            "Which audio clip best matches this number of repetitions? "
            "Choose A, B, C, or D."
        ),
        "conditions": {
            "modality": "Text",
            "input": f"{correct_text} times",
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
        "correct_answer": correct_answer,
    }
    
    return question
if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    
    DATASET_NAME = 'countixav'  # or 'vggss'
    OBJECT_NAME = 'action'
    # DATASET_NAME = 'URMP'  # or 'vggss'
    root_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/ExtremCountAV"
    mode = 'instances' # or instances
    N = 500
    
    export_dir = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/04_count/{DATASET_NAME}'
    if mode == 'classes':
        all_choices = os.listdir(root_dir)
    elif mode == 'instances':
        all_choices = os.listdir(root_dir)
        
        all_choices = [audio_name.split(".wav")[0] for audio_name in all_choices if audio_name.endswith('.wav')]   
        # all_choices = [audio_name for audio_name in all_choices if audio_name.endswith('.00')]   

    all_choices = [choice for choice in all_choices if os.path.exists(os.path.join(root_dir, f"{choice}.json"))]

    
    
    all_info = {}
    for video_id in all_choices:
        video_info = json.load(open(os.path.join(root_dir, f"{video_id}.json")))
        number_of_repetitions = video_info["number_of_repetitions"]
        if number_of_repetitions not in all_info:
            all_info[number_of_repetitions] = []
        all_info[number_of_repetitions].append(video_id)
        
    audio_text_questions = []
    text_audio_questions = []
    
    sorted_all_repetitions = sorted(all_info.keys())
    
    print(f"Total valid audio choices: {len(all_choices)}")
    
    for video_id in all_choices:
        video_info = json.load(open(os.path.join(root_dir, f"{video_id}.json")))
        number_of_repetitions = video_info["number_of_repetitions"]
        
        index_of_repetitions = sorted_all_repetitions.index(number_of_repetitions)
        
        choice_range = 10
        start_index = max(0, index_of_repetitions - choice_range)
        end_index = min(len(sorted_all_repetitions), index_of_repetitions + choice_range)
        
        candidate_choices = [i-index_of_repetitions for i in range(start_index, end_index, 2) if i != index_of_repetitions]
        random_add = random.sample(candidate_choices, 3)
        

        random_add = random_add + [0]
        random_add = random.sample(random_add, 4)
        correct_answer = ['A', 'B', 'C', 'D'][random_add.index(0)]
        
        audio_choices = []
        text_choices = []
        # import ipdb; ipdb.set_trace()
        for add in random_add:
            text_choices.append(sorted_all_repetitions[index_of_repetitions+add])
            if add == 0:
                audio_choices.append(video_id)
            else:
                number_of_repetitions_choice = sorted_all_repetitions[index_of_repetitions+add]
                if number_of_repetitions_choice in all_info:
                    video_ids_repetition = all_info[number_of_repetitions_choice]
                    video_id_choice = random.choice(video_ids_repetition)
                    audio_choices.append(video_id_choice)
                else:
                    raise ValueError(f"Number of repetitions {number_of_repetitions_choice} not found in all_info")
        
        audio_paths = []
        for audio_choice in audio_choices:
            audio_paths.append(os.path.join(root_dir, f"{audio_choice}.wav"))
      

        question_audio_text = \
            generate_question_audio_text(text_choices, audio_paths, correct_answer)
        
        question_text_audio = \
            generate_question_text_audio(text_choices, audio_paths, correct_answer)

        
        audio_text_questions.append(question_audio_text)
        text_audio_questions.append(question_text_audio)

    os.makedirs(export_dir, exist_ok=True)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_audio_text.json", "w") as f:
        json.dump(audio_text_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_text_audio.json", "w") as f:
        json.dump(text_audio_questions, f, indent=4)
    