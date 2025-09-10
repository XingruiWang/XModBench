import os
import re
import random
import json


# ==================== URMP Dataset Question Generator - 6 Modality Combinations ====================

# Mapping from instrument IDs to full names
URMP_id2class = {
    'vn': 'Violin',
    'vc': 'Cello', 
    'va': 'Viola',
    'fl': 'Flute',
    'cl': 'Clarinet',
    'tpt': 'Trumpet',
    'sax': 'Saxophone',
    'tbn': 'Trombone',
    'tba': 'Tuba',
    'ob': 'Oboe',
    'hn': 'French Horn',
    'db': 'Double Bass',
    'bn': 'Bassoon'
}


def parse_instruments_from_folder_name(folder_name):
    """
    Parse instrument combinations from folder name.
    Expected format: XX_YYY_inst1_inst2 (e.g., 01_Jupiter_vn_vc)
    Returns sorted tuple of instrument names for comparison.
    """
    # Split by underscore and get the instrument parts (last parts)
    parts = folder_name.split('_')
    
    # Look for instrument IDs in the parts
    instruments = []
    for part in parts:
        if part in URMP_id2class:
            instruments.append(part)
    
    # Return sorted tuple for easy comparison
    return tuple(sorted(instruments))


def get_instrument_names_string(folder_name):
    """
    Convert folder name to human-readable instrument combination string.
    Handles duplicates by using "N Instruments" format.
    """
    instrument_ids = parse_instruments_from_folder_name(folder_name)
    
    # Count occurrences of each instrument
    instrument_counts = {}
    for inst_id in instrument_ids:
        if inst_id in instrument_counts:
            instrument_counts[inst_id] += 1
        else:
            instrument_counts[inst_id] = 1
    
    # Convert to readable format with counts
    instrument_parts = []
    for inst_id, count in instrument_counts.items():
        instrument_name = URMP_id2class[inst_id]
        if count == 1:
            instrument_parts.append(instrument_name)
        else:
            # Pluralize the instrument name
            if instrument_name.endswith('s'):
                plural_name = instrument_name  # Already plural (like "French Horn" -> keep as is)
            elif instrument_name.endswith('y'):
                plural_name = instrument_name[:-1] + 'ies'  # e.g., "Timpany" -> "Timpanies"
            else:
                plural_name = instrument_name + 's'  # Most cases
            instrument_parts.append(f"{count} {plural_name}")
    
    # Sort for consistency
    instrument_parts.sort()
    
    if len(instrument_parts) == 1:
        return instrument_parts[0]
    elif len(instrument_parts) == 2:
        return f"{instrument_parts[0]} and {instrument_parts[1]}"
    elif len(instrument_parts) > 2:
        return ", ".join(instrument_parts[:-1]) + f", and {instrument_parts[-1]}"
    else:
        return folder_name  # Fallback to original name if no instruments found


def have_same_instrument_combination(folder_names):
    """
    Check if any folders have the same instrument combination.
    Returns True if there are duplicates, False otherwise.
    """
    instrument_combinations = [parse_instruments_from_folder_name(name) for name in folder_names]
    return len(set(instrument_combinations)) != len(instrument_combinations)


def generate_question_audio_vision(audio_choices, audio_paths, correct_answer, frame_in_folder=True, object_name='objects'):
    """Generate audio -> vision question: listen to sound and choose the matching image."""
    
    if frame_in_folder:
        image_frames_folders = [audio_sample.replace('.wav', '_frames') for audio_sample in audio_paths]
        image_paths = [os.path.join(folder, os.listdir(folder)[0]) for folder in image_frames_folders]
    else:
        image_paths = [audio_sample.replace('.wav', '.jpg') for audio_sample in audio_paths]    
    
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    
    # Convert audio choices to instrument names
    instrument_names = [get_instrument_names_string(choice) for choice in audio_choices]
    
    question = {
        "question": f"Which image most likely belongs to the {object_name} that makes this sound you hear? Answer with A, B, C, or D",
        "conditions": {
            "modality": "Audio",
            "input": correct_audio_sample,
        },
        "options": {
            "A": {"modality": "Image", "input": image_paths[0], "class": instrument_names[0]},
            "B": {"modality": "Image", "input": image_paths[1], "class": instrument_names[1]},
            "C": {"modality": "Image", "input": image_paths[2], "class": instrument_names[2]},
            "D": {"modality": "Image", "input": image_paths[3], "class": instrument_names[3]}
        },
        "correct_answer": correct_answer,
        "correct_class": instrument_names[ord(correct_answer) - ord('A')]
    }
    
    return question


def generate_question_vision_audio(audio_choices, audio_paths, correct_answer, frame_in_folder=True, object_name='objects'):
    """Generate vision -> audio question: look at image and choose the matching sound."""
    
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    
    if frame_in_folder:
        image_frames_folder = correct_audio_sample.replace('.wav', '_frames')
        image_path = os.path.join(image_frames_folder, os.listdir(image_frames_folder)[0])
    else:
        image_path = correct_audio_sample.replace('.wav', '.jpg')

    # Convert audio choices to instrument names
    instrument_names = [get_instrument_names_string(choice) for choice in audio_choices]
    
    question = {
        "question": f"Look at this image. Which audio clip is most likely produced by the {object_name} you see? Answer with A, B, C, or D",
        "conditions": {
            "modality": "Image",
            "input": image_path,
            "class": instrument_names[ord(correct_answer) - ord('A')]
        },
        "options": {
            "A": {"modality": "Audio", "input": audio_paths[0], "class": instrument_names[0]},
            "B": {"modality": "Audio", "input": audio_paths[1], "class": instrument_names[1]},
            "C": {"modality": "Audio", "input": audio_paths[2], "class": instrument_names[2]},
            "D": {"modality": "Audio", "input": audio_paths[3], "class": instrument_names[3]}
        },
        "correct_answer": correct_answer,
        "correct_class": instrument_names[ord(correct_answer) - ord('A')]
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
    
    # Convert audio choices to instrument names
    instrument_names = [get_instrument_names_string(choice) for choice in audio_choices]
    
    question = {
        "question": f"Look at this image. Which text description best matches the {object_name} you see? Answer with A, B, C, or D",
        "conditions": {
            "modality": "Image",
            "input": image_path,
            "class": instrument_names[correct_idx]
        },
        "options": {
            "A": {"modality": "Text", "input": instrument_names[0]},
            "B": {"modality": "Text", "input": instrument_names[1]},
            "C": {"modality": "Text", "input": instrument_names[2]},
            "D": {"modality": "Text", "input": instrument_names[3]}
        },
        "correct_answer": correct_answer,
        "correct_class": instrument_names[correct_idx]
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
    
    # Convert audio choices to instrument names
    instrument_names = [get_instrument_names_string(choice) for choice in audio_choices]
    correct_class = instrument_names[correct_idx]
    
    question = {
        "question": f"Based on this description: '{correct_class}', which image best matches the {object_name} described? Answer with A, B, C, or D",
        "conditions": {
            "modality": "Text",
            "input": correct_class,
        },
        "options": {
            "A": {"modality": "Image", "input": image_paths[0], "class": instrument_names[0]},
            "B": {"modality": "Image", "input": image_paths[1], "class": instrument_names[1]},
            "C": {"modality": "Image", "input": image_paths[2], "class": instrument_names[2]},
            "D": {"modality": "Image", "input": image_paths[3], "class": instrument_names[3]}
        },
        "correct_answer": correct_answer,
        "correct_class": correct_class
    }
    
    return question


def generate_question_text_audio(audio_choices, audio_paths, correct_answer, object_name='objects'):
    """Generate text -> audio question: read description and choose the matching sound."""
    
    correct_idx = ord(correct_answer) - ord('A')
    
    # Convert audio choices to instrument names
    instrument_names = [get_instrument_names_string(choice) for choice in audio_choices]
    correct_class = instrument_names[correct_idx]
    
    question = {
        "question": f"Based on this description: '{correct_class}', which audio clip best matches the {object_name} described? Answer with A, B, C, or D",
        "conditions": {
            "modality": "Text",
            "input": correct_class,
        },
        "options": {
            "A": {"modality": "Audio", "input": audio_paths[0], "class": instrument_names[0]},
            "B": {"modality": "Audio", "input": audio_paths[1], "class": instrument_names[1]},
            "C": {"modality": "Audio", "input": audio_paths[2], "class": instrument_names[2]},
            "D": {"modality": "Audio", "input": audio_paths[3], "class": instrument_names[3]}
        },
        "correct_answer": correct_answer,
        "correct_class": correct_class
    }
    
    return question


def generate_question_audio_text(audio_choices, audio_paths, correct_answer, object_name='objects'):
    """Generate audio -> text question: listen to sound and choose the matching description."""
    
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    correct_idx = ord(correct_answer) - ord('A')
    
    # Convert audio choices to instrument names
    instrument_names = [get_instrument_names_string(choice) for choice in audio_choices]
    
    question = {
        "question": f"Listen to this audio clip. Which text description best matches the {object_name} that makes this sound? Answer with A, B, C, or D",
        "conditions": {
            "modality": "Audio",
            "input": correct_audio_sample,
        },
        "options": {
            "A": {"modality": "Text", "input": instrument_names[0]},
            "B": {"modality": "Text", "input": instrument_names[1]},
            "C": {"modality": "Text", "input": instrument_names[2]},
            "D": {"modality": "Text", "input": instrument_names[3]}
        },
        "correct_answer": correct_answer,
        "correct_class": instrument_names[correct_idx]
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


def sample_unique_instrument_combinations(all_choices, n_samples=4, max_attempts=100):
    """
    Sample folders ensuring they have different instrument combinations.
    """
    for attempt in range(max_attempts):
        sampled_choices = random.sample(all_choices, n_samples)
        
        # Check if all have different instrument combinations
        if not have_same_instrument_combination(sampled_choices):
            return sampled_choices
    
    print(f"Warning: Could not find {n_samples} folders with different instrument combinations after {max_attempts} attempts")
    return None


def generate_all_modality_combinations(audio_choices, audio_paths, correct_answer, frame_in_folder=True, object_name='objects'):
    """
    Generate all 6 modality combinations for the given instances.
    """
    questions = {}
    
    questions['audio_vision'] = generate_question_audio_vision(audio_choices, audio_paths, correct_answer, frame_in_folder, object_name)
    questions['vision_audio'] = generate_question_vision_audio(audio_choices, audio_paths, correct_answer, frame_in_folder, object_name)
    questions['vision_text'] = generate_question_vision_text(audio_choices, audio_paths, correct_answer, frame_in_folder, object_name)
    questions['text_vision'] = generate_question_text_vision(audio_choices, audio_paths, correct_answer, frame_in_folder, object_name)
    questions['text_audio'] = generate_question_text_audio(audio_choices, audio_paths, correct_answer, object_name)
    questions['audio_text'] = generate_question_audio_text(audio_choices, audio_paths, correct_answer, object_name)
    
    return questions


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    
    # Configuration parameters
    DATASET_NAME = 'URMP'
    OBJECT_NAME = 'instruments composition'
    root_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/URMP_processed"
    mode = 'classes'  # or 'instances'
    frame_in_folder = False  # Set to True if images are in _frames folders
    N = 500  # Number of questions to generate
    
    # Select modality combinations to generate
    GENERATE_COMBINATIONS = [
        'audio_vision',  # Audio -> Vision
        'vision_audio',  # Vision -> Audio
        'vision_text',   # Vision -> Text
        'text_vision',   # Text -> Vision
        'text_audio',    # Text -> Audio
        'audio_text'     # Audio -> Text
    ]
    
    export_dir = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/instruments_comp_2'
    
    # Get all choices based on mode
    if mode == 'classes':
        all_choices = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and "Supplement" not in d]
    elif mode == 'instances':
        all_choices = [f for f in os.listdir(root_dir) if f.endswith('.wav')]
    
    print(f"Dataset: {DATASET_NAME}")
    print(f"Object type: {OBJECT_NAME}")
    print(f"Sampling mode: {mode}")
    print(f"Total valid choices: {len(all_choices)}")
    print(f"Frame in folder: {frame_in_folder}")
    print(f"Questions to generate: {N}")
    
    # Show some examples of instrument combinations
    print(f"\nSample instrument combinations:")
    for choice in all_choices[:5]:
        instruments = get_instrument_names_string(choice)
        print(f"  {choice} -> {instruments}")
    
    # Initialize question lists for each combination
    all_questions = {combo: [] for combo in GENERATE_COMBINATIONS}
    
    # Generate questions
    print(f"\nGenerating {N} questions for each modality combination...")
    successful_questions = 0
    
    for i in range(N):
        try:
            # Sample 4 choices with different instrument combinations
            audio_choices = sample_unique_instrument_combinations(all_choices, 4)
            
            if audio_choices is None:
                print(f"Skipping iteration {i+1}: could not find 4 different instrument combinations")
                continue
            
            # Get audio paths
            audio_paths = sample_instances(audio_choices, root_dir, mode=mode)
            
            # Skip if we couldn't get 4 valid audio paths
            if len(audio_paths) != 4:
                print(f"Skipping iteration {i+1}: only found {len(audio_paths)} valid audio files")
                continue
            
            # Random correct answer
            correct_answer = random.choice(['A', 'B', 'C', 'D'])
            
            # Generate questions for all modality combinations
            questions = generate_all_modality_combinations(
                audio_choices, audio_paths, correct_answer, 
                frame_in_folder=frame_in_folder, object_name=OBJECT_NAME
            )
            
            # Add to corresponding lists
            for combo in GENERATE_COMBINATIONS:
                if combo in questions:
                    all_questions[combo].append(questions[combo])
            
            successful_questions += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {successful_questions} questions ({i+1}/{N})")
                
        except Exception as e:
            print(f"Error in iteration {i+1}: {e}")
            continue
    
    print(f"Successfully generated {successful_questions} questions")
    
    # Save output files
    os.makedirs(export_dir, exist_ok=True)
    for combo in GENERATE_COMBINATIONS:
        filename = f"{export_dir}/{DATASET_NAME}_questions_{combo}.json"
        with open(filename, "w") as f:
            json.dump(all_questions[combo], f, indent=4)
        print(f"Saved {len(all_questions[combo])} questions to {filename}")
    
    # Save generation statistics
    stats = {
        "dataset": DATASET_NAME,
        "object_name": OBJECT_NAME,
        "sampling_mode": mode,
        "frame_in_folder": frame_in_folder,
        "total_choices_available": len(all_choices),
        "questions_requested": N,
        "questions_generated": successful_questions,
        "combinations": GENERATE_COMBINATIONS,
        "instrument_mapping": URMP_id2class,
        "choices_sample": all_choices[:10] if len(all_choices) > 10 else all_choices
    }
    
    with open(f"{export_dir}/{DATASET_NAME}_generation_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    print(f"\n=== Generation Summary ===")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Object type: {OBJECT_NAME}")
    print(f"Sampling mode: {mode}")
    print(f"Available choices: {len(all_choices)}")
    print(f"Questions requested: {N}")
    print(f"Questions generated: {successful_questions}")
    print(f"Generated combinations: {', '.join(GENERATE_COMBINATIONS)}")
    print(f"Export directory: {export_dir}")