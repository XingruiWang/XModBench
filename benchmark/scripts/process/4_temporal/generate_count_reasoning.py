import os
import re
import random
import json
import numpy as np
def generate_question_vision_audio(text_choices, audio_paths, correct_answer, vision_modality='Image', formula_text="", correct_instance=None, correct_repetition=None):
    """
    Generate a question from vision to text.
    Given an image, ask which text description most likely corresponds to their spatial arrangement from left to right.
    """
    correct_audio_path = correct_instance
    
    if vision_modality == 'Image':
        correct_image_path = correct_audio_path.replace(".wav", ".png")
        question_text = ''.join([
            "Based on the frequency spectrum image of sound of repeated actions, "
            "select the audio that matches the repetition count. "
            f"Then calculate: {formula_text} "
            "Which audio clip best matches the resulting number? Choose A, B, C, or D."
        ])

    elif vision_modality == 'Video':
        correct_image_path = correct_audio_path.replace(".wav", ".mp4")
        question_text = ''.join([
            "Based on the video of repeated actions, "
            "select the audio that matches the repetition count. "
            f"Then calculate: {formula_text} "
            "Which audio clip best matches the resulting number? Choose A, B, C, or D."
        ])
    question = {
        "question": question_text,
        "conditions": {
            "modality": vision_modality,
            "input": correct_image_path,
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

def generate_question_audio_vision(text_choices, audio_paths, correct_answer, vision_modality='Image', formula_text="", correct_instance=None, correct_repetition=None):
    """
    Generate a question from vision to text.
    Given an image, ask which text description most likely corresponds to their spatial arrangement from left to right.
    """
    
    correct_audio_path = correct_instance

    if vision_modality == 'Image':
        image_paths = [audio_path.replace(".wav", ".png") for audio_path in audio_paths]
        question_text = ''.join([
            "Listen carefully to the audio clip and count how many times the action is repeated. "
            f"Then calculate the result using this formula: {formula_text}. "
            "Which frequency spectrum image corresponds to that final number of repetitions? Choose A, B, C, or D."
        ])

    elif vision_modality == 'Video':
        image_paths = [audio_path.replace(".wav", ".mp4") for audio_path in audio_paths]
        question_text = ''.join([
            "Listen carefully to the audio clip and determine how many times the action is repeated. "
            f"Then apply the following formula: {formula_text}. "
            "Which video of repeated actions matches the resulting number of repetitions? Choose A, B, C, or D."
        ])
        
    question = {
        "question": question_text,
        "conditions": {
            "modality": "Audio",
            "input": correct_audio_path,
        },
        "options": {
            "A": {
                "modality": vision_modality,
                "input": image_paths[0],
            },
            "B": {
                "modality": vision_modality,
                "input": image_paths[1],
            },
            "C": {
                "modality": vision_modality,
                "input": image_paths[2],
            },
            "D": {
                "modality": vision_modality,
                "input": image_paths[3],
            }
        },
        "correct_answer": correct_answer,
    }
    
    return question
    
def generate_question_vision_text(text_choices, audio_paths, correct_answer, vision_modality='Image', formula_text="", correct_instance=None, correct_repetition=None):
    """
    Generate a question from vision to text.
    Given an image, ask which text description most likely corresponds to their spatial arrangement from left to right.
    """
    correct_audio_path = correct_instance
    
    if vision_modality == 'Image':
        correct_image_path = correct_audio_path.replace(".wav", ".png")
        question_text = ''.join([
            "Given the frequency spectrum image representing repeated actions, "
            "first estimate how many times the action is repeated. "
            f"Then apply the following formula: {formula_text}. "
            "Which text correctly describes the resulting number? Choose A, B, C, or D."
        ])

    elif vision_modality == 'Video':
        correct_image_path = correct_audio_path.replace(".wav", ".mp4")
        question_text = ''.join([
            "The video shows a repeated motion. "
            "Start by counting how many times the action is repeated. "
            f"Then calculate the result using this formula: {formula_text}. "
            "Which text option correctly matches that number? Choose A, B, C, or D."
        ])
        
    question = {
        "question": question_text,
        "conditions": {
            "modality": vision_modality,
            "input": correct_image_path,
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

def generate_question_text_vision(text_choices, audio_paths, correct_answer, vision_modality='Image', formula_text="", correct_instance=None, correct_repetition=None):
    """
    Generate a question from text to vision.
    Given a text description of the number of repetitions of an action, ask which image most likely corresponds to the number of repetitions.
    """
    correct_text = correct_repetition
    correct_audio_path = correct_instance
    
    if vision_modality == 'Image':
        image_paths = [audio_path.replace(".wav", ".png") for audio_path in audio_paths]
        question_text = ''.join([
            "The text describes how many times an action is repeated. "
            f"Apply the following formula to compute the final result: {formula_text}. "
            "Which frequency spectrum image shows the same number of repeated times as that result? Choose A, B, C, or D."
        ])

    elif vision_modality == 'Video':
        image_paths = [audio_path.replace(".wav", ".mp4") for audio_path in audio_paths]
        question_text = ''.join([
            "The text specifies a repetition count. "
            f"Use this formula to compute the final result: {formula_text}. "
            "Which video of repeated actions shows the same number of repeated times as that result? Choose A, B, C, or D."
        ])
        
    question = {
        "question": question_text,
        "conditions": {
            "modality": "Text",
            "input": f"{correct_text} times",
        },
        "options": {
            "A": {
                "modality": vision_modality,
                "input": image_paths[0],
            },
            "B": {
                "modality": vision_modality,
                "input": image_paths[1],
            },
            "C": {
                "modality": vision_modality,
                "input": image_paths[2],
            },
            "D": {
                "modality": vision_modality,
                "input": image_paths[3],
            }
        },
        "correct_answer": correct_answer,
    }
    
    return question

def generate_question_audio_text(text_choices, audio_paths, correct_answer, formula_text="", correct_instance=None, correct_repetition=None):
    """
    Generate a question from audio to text.
    Given an audio clip, ask which text description most likely corresponds to the spatial arrangement.
    """

    correct_audio_path = correct_instance
    
    question = {
        "question": ''.join([
            "Listen carefully to the audio clip and count how many times the action is repeated. "
            f"Then apply the following formula to compute the final answer: {formula_text}. "
            "Which text best describes the resulting number? Choose A, B, C, or D."
        ]),
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

def generate_question_text_audio(text_choices, audio_paths, correct_answer,  formula_text="", correct_instance=None, correct_repetition=None):
    """
    Generate a question from text to audio.
    Given a text description, ask which audio clip most likely corresponds to the spatial arrangement.
    """

    correct_text = correct_repetition
    
    question = {
        "question": ''.join([
            f"Apply the following formula to this number of repetitions: {formula_text}. "
            "Then listen carefully to each audio clip—focus on how many repetitions of the action are in the audio. "
            "Which audio clip best matches the resulting number? "
            "Choose A, B, C, or D."
        ]),
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

import random

def generate_math_question(correct_repetition, all_canditates):
    """
    Generate a math formula, apply it to the repetition count (correct answer),
    and return:
        - new_choices: list of 4 numeric answer options (shuffled)
        - correct_answer: correct result after formula
        - formula_text: text to insert into prompt
    """
    # Step 2: Randomly pick an operation type
    trials = 0
    MAX_TRIALS = 100
    while trials < MAX_TRIALS:
        trials += 1
        op_type = random.choice(["mul", "add_mul", "sub_mul"])
        # if op_type == "add":
        #     k = random.randint(1, 5)
        #     result = correct_repetition + k
        #     formula_text = f"(repetition count + {k})"
        # elif op_type == "sub":
        #     k = random.randint(1, min(5, correct_repetition - 1)) if correct_repetition > 1 else 1
        #     result = correct_repetition - k
        #     formula_text = f"(repetition count - {k})"
        if op_type == "mul":
            k = random.randint(2, 4)
            result = correct_repetition * k
            formula_text = f"(repetition count * {k})"
        elif op_type == "add_mul":
            a = random.randint(1, 3)
            b = random.randint(2, 4)
            result = (correct_repetition + a) * b
            formula_text = f"({correct_repetition} + {a}) * {b}"
            formula_text = f"((repetition count + {a}) * {b})"
        elif op_type == "sub_mul":
            a = random.randint(1, min(3, correct_repetition - 1)) if correct_repetition > 1 else 1
            b = random.randint(2, 4)
            result = (correct_repetition - a) * b
            formula_text = f"((repetition count - {a}) * {b})"
            if result < 0:
                # Ensure result is non-negative
                result = -result
                formula_text = f"(({a} - repetition count) * {b})"
    
        if result in all_canditates:
            break
        
    if trials >= MAX_TRIALS:
        target = random.choice(all_canditates)
        
        k = result - target
        
        if k >= 0:
            formula_text = f"({formula_text} - {k})"
        else:
            k = -k
            formula_text = f"({formula_text} + {k})"
        result = target

    # Step 3: Generate 3 distractor options
    return result, formula_text

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    
    DATASET_NAME = 'countixav'  # or 'vggss'
    OBJECT_NAME = 'action'
    # DATASET_NAME = 'URMP'  # or 'vggss'
    root_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/ExtremCountAV"
    mode = 'instances' # or instances
    vision_modality = 'Video'  # or 'Video'
    # vision_modality = 'Video'  # or 'Video'
    N = 500
    
    export_dir = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/04_temporal/count_calculation/'
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
        
    max_repetitions = max(all_info.keys())
    min_repetitions = min(all_info.keys())
    
    audio_text_questions = []
    text_audio_questions = []
    text_vision_questions = []
    vision_text_questions = []
    audio_vision_questions = []
    vision_audio_questions = []
    
    sorted_all_repetitions = sorted(all_info.keys())
    
    print(f"Total valid audio choices: {len(all_choices)}")
    
    for video_id in all_choices:
        video_info = json.load(open(os.path.join(root_dir, f"{video_id}.json")))
        action_class = video_info["action_class"]

        number_of_repetitions = video_info["number_of_repetitions"]
        
        cal_result, formula_text = generate_math_question(number_of_repetitions, list(all_info.keys()))
        
        if number_of_repetitions > 10:
            continue
        index_of_repetitions = sorted_all_repetitions.index(cal_result)
        
        choice_range = 10
        start_index = max(0, index_of_repetitions - choice_range)
        end_index = min(len(sorted_all_repetitions), index_of_repetitions + choice_range)
        
        candidate_choices = [i-index_of_repetitions for i in range(start_index, end_index, 5) if np.abs(i - index_of_repetitions) > 1]
        
        if len(candidate_choices) < 3:
            candidate_choices += random.sample([i-index_of_repetitions for i in range(start_index, end_index) if np.abs(i - index_of_repetitions) > 5 and i not in candidate_choices], 3 - len(candidate_choices))
        
        random_add = random.sample(candidate_choices, 3)

        random_add = random_add + [0]
        random_add = random.sample(random_add, 4)
        correct_answer = ['A', 'B', 'C', 'D'][random_add.index(0)]
        
        audio_choices = []
        text_choices = []
        # import ipdb; ipdb.set_trace()
        for add in random_add:

            text_choices.append(sorted_all_repetitions[index_of_repetitions+add])

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
        
        correct_instance = os.path.join(root_dir, f"{video_id}.wav")
        
        question_audio_text = \
            generate_question_audio_text(text_choices, audio_paths, correct_answer, formula_text=formula_text, correct_instance=correct_instance, correct_repetition=number_of_repetitions)
        
        question_text_audio = \
            generate_question_text_audio(text_choices, audio_paths, correct_answer, formula_text=formula_text, correct_instance=correct_instance, correct_repetition=number_of_repetitions)
        
        question_text_vision = \
            generate_question_text_vision(text_choices, audio_paths, correct_answer, vision_modality=vision_modality, formula_text=formula_text, correct_instance=correct_instance, correct_repetition=number_of_repetitions)
            
        question_vidion_text = \
            generate_question_vision_text(text_choices, audio_paths, correct_answer, vision_modality=vision_modality, formula_text=formula_text, correct_instance=correct_instance, correct_repetition=number_of_repetitions)
            
        question_audio_vision = \
            generate_question_audio_vision(text_choices, audio_paths, correct_answer, vision_modality=vision_modality, formula_text=formula_text, correct_instance=correct_instance, correct_repetition=number_of_repetitions)
            
        question_vision_audio = \
            generate_question_vision_audio(text_choices, audio_paths, correct_answer, vision_modality=vision_modality, formula_text=formula_text, correct_instance=correct_instance, correct_repetition=number_of_repetitions)
        
        
        audio_text_questions.append(question_audio_text)
        text_audio_questions.append(question_text_audio)
        text_vision_questions.append(question_text_vision)
        vision_text_questions.append(question_vidion_text)
        audio_vision_questions.append(question_audio_vision)
        vision_audio_questions.append(question_vision_audio)
        
        

    os.makedirs(export_dir, exist_ok=True)
   
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_audio_text.json", "w") as f:
        json.dump(audio_text_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_text_audio.json", "w") as f:
        json.dump(text_audio_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_text_vision.json", "w") as f:
        json.dump(text_vision_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_vision_text.json", "w") as f:
        json.dump(vision_text_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_audio_vision.json", "w") as f:
        json.dump(audio_vision_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_vision_audio.json", "w") as f:
        json.dump(vision_audio_questions, f, indent=4)
        
        
    print(f"Exported {len(audio_text_questions)} audio-text questions and {len(text_audio_questions)} text-audio questions and {len(text_vision_questions)} text-vision questions")
    