import os
import re
import random
import json


# ==================== 1. Main function: Load model and answer one question ====================

def load_description(class_name, direction):
    if class_name == -1 or class_name == "-1":
        return "There is no vehicle moving in the scene."
    
    return f"A {class_name} moving {direction}."

def generate_question_audio_vision(all_choices, correct_answer, object_name='objects'):

    correct_choice = all_choices[ord(correct_answer) - ord('A')]
    other_choices = [all_choices[i] for i in range(4) if i != ord(correct_answer) - ord('A')]
    
    descriptions = [load_description(choice['label'], choice['direction']) for choice in all_choices]
    correct_description = descriptions[ord(correct_answer) - ord('A')]
    
    image_paths = [choice['image_path'] for choice in all_choices]
    
    audio_paths = [choice['audio_path'] for choice in all_choices]
    
    question = {
        "question" : "Based on the spatial audio clip, try to imagine what the urban scene looks like—consider which direction the sound is coming from and how close the vehicle gets. Then select the video that best matches the expected position and type of vehicle. Choose A, B, C, or D.",

        "conditions": {
            "modality": "Audio",
            "input": correct_choice['audio_path'],
            "description": correct_description,
            },
        "options": {
            "A": {
                "modality": "Video",
                "input": image_paths[0],
                "description": descriptions[0],
            },
            "B": {
                "modality": "Video",
                "input": image_paths[1],
                "description": descriptions[1],
            },
            "C": {
                "modality": "Video",
                "input": image_paths[2],
                "description": descriptions[2],
            },
            "D": {
                "modality": "Video",
                "input": image_paths[3],
                "description": descriptions[3],
            }
        },
        "correct_answer": correct_answer
    }
    
    return question


def generate_question_vision_audio(all_choices, correct_answer, object_name='objects'):
    
    correct_choice = all_choices[ord(correct_answer) - ord('A')]
    other_choices = [all_choices[i] for i in range(4) if i != ord(correct_answer) - ord('A')]
    
    descriptions = [load_description(choice['label'], choice['direction']) for choice in all_choices]
    correct_description = descriptions[ord(correct_answer) - ord('A')]
    image_paths = [choice['image_path'] for choice in all_choices]
    audio_paths = [choice['audio_path'] for choice in all_choices]
    
    question = {
       "question": "Carefully observe the video clip of the urban scene. Pay attention to the vehicle's position and motion direction (e.g., moving toward the camera or from left to right). Then select the audio clip that best matches the expected sound of that movement and direction. Choose A, B, C, or D.",
        "conditions": {
            "modality": "Video",
            "input":  correct_choice['image_path'],
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
        "correct_answer": correct_answer
    }
    
    return question 
    
def generate_question_vision_text(all_choices, correct_answer, object_name='objects'):
    """
    Generate a question from vision to text.
    Given an image, ask which text description most likely corresponds to their spatial arrangement from left to right.
    """
    correct_choice = all_choices[ord(correct_answer) - ord('A')]
    other_choices = [all_choices[i] for i in range(4) if i != ord(correct_answer) - ord('A')]
    
    descriptions = [load_description(choice['label'], choice['direction']) for choice in all_choices]
    correct_description = descriptions[ord(correct_answer) - ord('A')]
    image_paths = [choice['image_path'] for choice in all_choices]
    audio_paths = [choice['audio_path'] for choice in all_choices]
    
    question = {
        "question": "Look closely at the video clip of the urban scene, including the type of vehicle, its size, position, and direction of motion. Based on this observation, which text best describes the vehicle's behavior in the scene? Choose A, B, C, or D.",
        "conditions": {
            "modality": "Video",
            "input":  correct_choice['image_path'],
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
        "correct_answer": correct_answer
    }
    
    return question

def generate_question_text_vision(all_choices, correct_answer, object_name='objects'):
    """
    Generate a question from text to vision.
    Given a text description of the spatial arrangement of instruments from left to right, ask which video clip most likely corresponds to the spatial arrangement.
    """
    correct_choice = all_choices[ord(correct_answer) - ord('A')]
    other_choices = [all_choices[i] for i in range(4) if i != ord(correct_answer) - ord('A')]
    
    descriptions = [load_description(choice['label'], choice['direction']) for choice in all_choices]
    correct_description = descriptions[ord(correct_answer) - ord('A')]
    image_paths = [choice['image_path'] for choice in all_choices]
    audio_paths = [choice['audio_path'] for choice in all_choices]
    
    question = {
        "question": "The text describes an event in an urban street scene, including the vehicle type and movement direction (e.g., a car approaching from the right). Which video clip best illustrates that situation? Choose A, B, C, or D.",
        "conditions": {
            "modality": "Text",
            "input": correct_description,
            "description": correct_description,
        },
        "options": {
            "A": {
                "modality": "Video",
                "input": image_paths[0],
                "description": descriptions[0],
            },
            "B": {
                "modality": "Video",
                "input": image_paths[1],
                "description": descriptions[1],
            },
            "C": {
                "modality": "Video",
                "input": image_paths[2],
                "description": descriptions[2],
            },
            "D": {
                "modality": "Video",
                "input": image_paths[3],
                "description": descriptions[3],
            }
        },
        "correct_answer": correct_answer
    }
    
    return question

def generate_question_audio_text(all_choices, correct_answer, object_name='objects'):
    """
    Generate a question from audio to text.
    Given an audio clip, ask which text description most likely corresponds to the spatial arrangement.
    """
    correct_choice = all_choices[ord(correct_answer) - ord('A')]
    other_choices = [all_choices[i] for i in range(4) if i != ord(correct_answer) - ord('A')]
    
    descriptions = [load_description(choice['label'], choice['direction']) for choice in all_choices]
    correct_description = descriptions[ord(correct_answer) - ord('A')]
    image_paths = [choice['image_path'] for choice in all_choices]
    audio_paths = [choice['audio_path'] for choice in all_choices]
    
    question = {
        "question": "Listen carefully to the spatial audio from an urban scene. Consider the vehicle type (if any), its movement direction, and distance. Which text description best matches what you hear? Choose A, B, C, or D.",
        "conditions": {
            "modality": "Audio",
            "input": correct_choice['audio_path'],
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
        "correct_answer": correct_answer
    }
    
    return question

def generate_question_text_audio(all_choices, correct_answer, object_name='objects'):
    """
    Generate a question from text to audio.
    Given a text description, ask which audio clip most likely corresponds to the spatial arrangement.
    """
    correct_choice = all_choices[ord(correct_answer) - ord('A')]
    other_choices = [all_choices[i] for i in range(4) if i != ord(correct_answer) - ord('A')]
    
    descriptions = [load_description(choice['label'], choice['direction']) for choice in all_choices]
    correct_description = descriptions[ord(correct_answer) - ord('A')]
    image_paths = [choice['image_path'] for choice in all_choices]
    audio_paths = [choice['audio_path'] for choice in all_choices]
    
    question = {
        "question": "Read the description of the vehicle movement in the urban scene, including direction and distance. Which audio clip best represents how that movement would sound in space? Choose A, B, C, or D.",
        "conditions": {
            "modality": "Text",
            "input": correct_description,
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
        "correct_answer": correct_answer
    }
    
    return question



if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    
    DATASET_NAME = 'urbansas'  # or 'vggss'
    OBJECT_NAME = 'objects'

    root_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/urbansas_samples_videos_filtered"
    
    N = 0
    
    export_dir = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/02_spatial/3D_movements'

    audio_vision_questions = []
    vision_audio_questions = []
    vision_text_questions = []
    text_vision_questions = []
    text_audio_questions = []
    audio_text_questions = []
     
    meta_data = "/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/process/2_spatial_audio/movements/urbansas_extracted_samples.csv"
    # read metadata csv
    with open(meta_data, 'r') as f:
        lines = f.readlines()
        lines = [line.split(',"[')[0].strip(',') for line in lines if line.strip()]
        all_metadata = [re.split(r',\s*', line) for line in lines[1:]]
    all_metadata_dict = {
        "empty": [],
        "car": {
            "left_to_right": [],
            "right_to_left": [],
            "close_to_far": [],
            "far_to_close": []
        },
        "bus": {
            "left_to_right": [],
            "right_to_left": [],
            "close_to_far": [],
            "far_to_close": []
        }
    }

    id2class = {
        0: "car",
        1: "bus",
        2: "motorbike",
        3: "truck",
        4: "offscene"
    }
    for metadata in all_metadata:
        try:
            clip_id,class_id,label,start,end,direction_x,direction_y,audio_clip,video_clip = metadata
        except:
            continue
        image_path = audio_clip.replace('.wav', '.mp4')
        class_id = int(class_id)
        if class_id == -1:
            
            all_metadata_dict["empty"].append({
                "clip_id": clip_id,
                "label": label,
                "start": start,
                "end": end,
                "direction_x": 0,
                "direction_y": 0,
                "audio_path": audio_clip,
                "image_path": image_path,
                "direction": "empty"
            })
            continue
        
        class_name = id2class[int(class_id)]
        if class_name not in all_metadata_dict:
            all_metadata_dict[class_name] = {
                "left_to_right": [],
                "right_to_left": [],
                "close_to_far": [],
                "far_to_close": []
            }
        
        if label != 4:  # not offscene
            if direction_x == '':
                continue
            direction_x = float(direction_x)
            direction_y = float(direction_y)
            if direction_x > 100:
                all_metadata_dict[class_name]["left_to_right"].append({
                        "clip_id": clip_id,
                        "label": label,
                        "start": start,
                        "end": end,
                        "direction_x": direction_x,
                        "direction_y": direction_y,
                        "audio_path": audio_clip,
                        "image_path": image_path,
                        "direction": "from left to right"
                    })
            elif direction_x < -100:
                all_metadata_dict[class_name]["right_to_left"].append({
                        "clip_id": clip_id,
                        "label": label,
                        "start": start,
                        "end": end,
                        "direction_x": direction_x,
                        "direction_y": direction_y,
                        "audio_path": audio_clip,
                        "image_path": image_path,
                        "direction": "from right to left"
                    })
            if direction_y > 100:
                all_metadata_dict[class_name]["close_to_far"].append({
                        "clip_id": clip_id,
                        "label": label,
                        "start": start,
                        "end": end,
                        "direction_x": direction_x,
                        "direction_y": direction_y,
                        "audio_path": audio_clip,
                        "image_path": image_path,
                        "direction": "from close to far"
                    })
            elif direction_y < -100:
                all_metadata_dict[class_name]["far_to_close"].append({
                        "clip_id": clip_id,
                        "label": label,
                        "start": start,
                        "end": end,
                        "direction_x": direction_x,
                        "direction_y": direction_y,
                        "audio_path": audio_clip,
                        "image_path": image_path,
                        "direction": "from far to close"
                    })
    
    all_class_names = set(all_metadata_dict.keys()) - {"empty", "offscene"}
    for class_name in all_metadata_dict:
        if class_name == "empty" or class_name == "offscene":
            continue
        for direction in all_metadata_dict[class_name]:
            # Sample 1: same class but opposite direction
            # Sample 2: different class 
            # empty scene
            for sample in all_metadata_dict[class_name][direction]:

                if direction == "left_to_right":
                    opposite_direction = "right_to_left"
                elif direction == "right_to_left":
                    opposite_direction = "left_to_right"
                elif direction == "close_to_far":
                    opposite_direction = "far_to_close"
                elif direction == "far_to_close":
                    opposite_direction = "close_to_far"
                    
                try:
                    sample_1 = random.choice(all_metadata_dict[class_name][opposite_direction])
                except IndexError:
                    sample_1 = random.choice(all_metadata_dict['empty'])

                
                sample_2_class = random.choice(list(all_class_names - {class_name}))
                sample_2_direction = random.choice(list(all_metadata_dict[sample_2_class].keys()))
                try:
                    sample_2 = random.choice(all_metadata_dict[sample_2_class][sample_2_direction])
                except IndexError:
                    sample_2 = random.choice(all_metadata_dict['empty'])
                    
                sample_3 = random.choice(all_metadata_dict['empty'])

                choises = [sample, sample_1, sample_2, sample_3]
            
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
                  
                
                
                
        
        
        
        