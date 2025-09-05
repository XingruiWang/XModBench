import os
import re
import random
import json


# ==================== 1. Main function: Load model and answer one question ====================

def load_description(class_name, direction):
    if class_name == -1 or class_name == "-1":
        return "No vehicle is present in the scene."
    
    # More descriptive vehicle movement descriptions
    direction_mapping = {
        "from left to right": "moving horizontally from left to right across the scene",
        "from right to left": "moving horizontally from right to left across the scene", 
        "from close to far": "moving away from the camera/observer into the distance",
        "from far to close": "approaching toward the camera/observer from the distance",
        "empty": "no movement detected"
    }
    
    enhanced_direction = direction_mapping.get(direction, direction)
    return f"A {class_name} {enhanced_direction}."

def generate_question_audio_vision(all_choices, correct_answer, object_name='objects'):
    correct_choice = all_choices[ord(correct_answer) - ord('A')]
    other_choices = [all_choices[i] for i in range(4) if i != ord(correct_answer) - ord('A')]
    
    descriptions = [load_description(choice['label'], choice['direction']) for choice in all_choices]
    correct_description = descriptions[ord(correct_answer) - ord('A')]
    
    image_paths = [choice['image_path'] for choice in all_choices]
    audio_paths = [choice['audio_path'] for choice in all_choices]
    
    question = {
        "question": "Listen carefully to the spatial audio clip from an urban street scene. Pay attention to:\n- The type of vehicle sound (car, bus, motorbike, or no vehicle)\n- The direction of movement (left-to-right, right-to-left, approaching, or receding)\n- The spatial positioning and distance changes\n\nBased on these audio cues, select the video that best matches the vehicle type and movement pattern you heard. Choose A, B, C, or D.",

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
       "question": "Analyze the video clip of the urban street scene carefully. Identify:\n- The type of vehicle shown (car, bus, motorbike, or no vehicle)\n- The vehicle's movement direction and trajectory\n- How the vehicle's position changes relative to the camera\n\nBased on your visual analysis, select the audio clip that would realistically correspond to this vehicle's movement through 3D space. Consider how the sound would change as the vehicle moves. Choose A, B, C, or D.",
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
    """
    correct_choice = all_choices[ord(correct_answer) - ord('A')]
    other_choices = [all_choices[i] for i in range(4) if i != ord(correct_answer) - ord('A')]
    
    descriptions = [load_description(choice['label'], choice['direction']) for choice in all_choices]
    correct_description = descriptions[ord(correct_answer) - ord('A')]
    image_paths = [choice['image_path'] for choice in all_choices]
    audio_paths = [choice['audio_path'] for choice in all_choices]
    
    question = {
        "question": "Examine the video clip of the urban street scene. Focus on:\n- Identifying the specific type of vehicle (car, bus, motorbike) or if no vehicle is present\n- Determining the vehicle's movement pattern and direction\n- Observing how the vehicle's size and position change over time\n\nWhich text description most accurately captures what you observed in the video? Choose A, B, C, or D.",
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
    """
    correct_choice = all_choices[ord(correct_answer) - ord('A')]
    other_choices = [all_choices[i] for i in range(4) if i != ord(correct_answer) - ord('A')]
    
    descriptions = [load_description(choice['label'], choice['direction']) for choice in all_choices]
    correct_description = descriptions[ord(correct_answer) - ord('A')]
    image_paths = [choice['image_path'] for choice in all_choices]
    audio_paths = [choice['audio_path'] for choice in all_choices]
    
    question = {
        "question": "Read the text description of the urban street scene, which specifies:\n- The type of vehicle involved (or absence of vehicles)\n- The movement direction and pattern\n- The spatial trajectory through the scene\n\nBased on this description, identify which video clip visually demonstrates the described scenario. Look for matching vehicle type and movement pattern. Choose A, B, C, or D.",
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
    """
    correct_choice = all_choices[ord(correct_answer) - ord('A')]
    other_choices = [all_choices[i] for i in range(4) if i != ord(correct_answer) - ord('A')]
    
    descriptions = [load_description(choice['label'], choice['direction']) for choice in all_choices]
    correct_description = descriptions[ord(correct_answer) - ord('A')]
    image_paths = [choice['image_path'] for choice in all_choices]
    audio_paths = [choice['audio_path'] for choice in all_choices]
    
    question = {
        "question": "Listen to the spatial audio from the urban street scene. Analyze:\n- The vehicle engine sound characteristics to identify the vehicle type (car, bus, motorbike)\n- The audio's directional movement (left-to-right, right-to-left, approaching, receding)\n- Volume and frequency changes that indicate distance and movement\n- Whether any vehicle sound is present at all\n\nWhich text description best matches your audio analysis? Choose A, B, C, or D.",
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
    """
    correct_choice = all_choices[ord(correct_answer) - ord('A')]
    other_choices = [all_choices[i] for i in range(4) if i != ord(correct_answer) - ord('A')]
    
    descriptions = [load_description(choice['label'], choice['direction']) for choice in all_choices]
    correct_description = descriptions[ord(correct_answer) - ord('A')]
    image_paths = [choice['image_path'] for choice in all_choices]
    audio_paths = [choice['audio_path'] for choice in all_choices]
    
    question = {
        "question": "Read the description of the vehicle movement in the urban scene. It specifies:\n- The vehicle type (car, bus, motorbike, or no vehicle)\n- The direction and pattern of movement\n- The spatial relationship to the observer\n\nBased on this information, predict how this scenario would sound in real 3D space. Which audio clip best represents the described movement with appropriate spatial audio cues? Choose A, B, C, or D.",
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


# Additional helper function to improve choice generation
def generate_better_distractors(sample, all_metadata_dict, all_class_names):
    """Generate more strategic distractors for better evaluation"""
    class_name = sample.get('label', 'empty')
    direction = sample['direction']
    
    distractors = []
    
    # Distractor 1: Same vehicle type, opposite direction
    if direction in ["from left to right", "from right to left"]:
        opposite_direction = "from right to left" if direction == "from left to right" else "from left to right"
    elif direction in ["from close to far", "from far to close"]:
        opposite_direction = "from far to close" if direction == "from close to far" else "from close to far"
    else:
        opposite_direction = "empty"
    
    try:
        if class_name != "empty" and opposite_direction in all_metadata_dict[class_name]:
            distractor_1 = random.choice(all_metadata_dict[class_name][opposite_direction])
        else:
            distractor_1 = random.choice(all_metadata_dict['empty'])
    except (KeyError, IndexError):
        distractor_1 = random.choice(all_metadata_dict['empty'])
    
    # Distractor 2: Different vehicle type, same direction
    try:
        other_classes = list(all_class_names - {class_name})
        if other_classes and class_name != "empty":
            other_class = random.choice(other_classes)
            if direction in all_metadata_dict[other_class]:
                distractor_2 = random.choice(all_metadata_dict[other_class][direction])
            else:
                distractor_2 = random.choice(all_metadata_dict['empty'])
        else:
            distractor_2 = random.choice(all_metadata_dict['empty'])
    except (KeyError, IndexError):
        distractor_2 = random.choice(all_metadata_dict['empty'])
    
    # Distractor 3: Empty scene or random other scenario
    distractor_3 = random.choice(all_metadata_dict['empty'])
    
    return [distractor_1, distractor_2, distractor_3]


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
            for sample in all_metadata_dict[class_name][direction]:
                # Use improved distractor generation
                distractors = generate_better_distractors(sample, all_metadata_dict, all_class_names)
                
                choices = [sample] + distractors
            
                # Randomize order
                order = [0, 1, 2, 3]
                order = random.sample(order, 4)
                
                correct_answer = ['A', 'B', 'C', 'D'][order.index(0)]
                
                choices = [choices[i] for i in order]
                
                question_audio_vision = generate_question_audio_vision(choices, correct_answer)
                question_vision_audio = generate_question_vision_audio(choices, correct_answer)
                question_vision_text = generate_question_vision_text(choices, correct_answer)
                question_text_vision = generate_question_text_vision(choices, correct_answer)
                question_audio_text = generate_question_audio_text(choices, correct_answer)
                question_text_audio = generate_question_text_audio(choices, correct_answer)
                
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