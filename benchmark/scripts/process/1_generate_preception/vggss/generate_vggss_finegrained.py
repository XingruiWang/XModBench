import os
import re
import random
import json


# ==================== Multimodal Question-Answer Generator with Category Constraints ====================


def load_blacklist(blacklist_file):
    """
    Load low-quality instance blacklist from file.
    
    Args:
        blacklist_file (str): Path to the blacklist file containing instance names
        
    Returns:
        set: Set of blacklisted instance names (without _frames suffix)
    """
    blacklist = set()
    if os.path.exists(blacklist_file):
        with open(blacklist_file, 'r') as f:
            for line in f:
                # Remove _frames suffix and keep only the file name
                if '_frames' in line:
                    instance_name = line.strip().replace('_frames', '')
                else:
                    instance_name = line.strip().replace('.wav', '')
                if instance_name:
                    blacklist.add(instance_name)
    return blacklist


def load_category_data(category_file, blacklist=None):
    """
    Load category data and group instances by category, filtering out blacklisted items.
    
    Args:
        category_file (str): Path to the JSON file containing category information
        blacklist (set): Set of blacklisted instance names to filter out
        
    Returns:
        dict: Dictionary with categories as keys and list of instances as values
    """
    if blacklist is None:
        blacklist = set()
    
    with open(category_file, 'r') as f:
        data = json.load(f)
    
    # Filter blacklisted instances and group by category
    category_groups = {}
    for item in data:
        file_name = item['file']
        
        # Check if instance is blacklisted
        if file_name in blacklist:
            continue
            
        category = item['category']
        if category not in category_groups:
            category_groups[category] = []
        
        category_groups[category].append({
            'file': file_name,
            'class': item['class'],
            'category': category,
            'bbox': item.get('bbox', [])
        })
    
    return category_groups


def generate_question_audio_vision(instances, correct_answer, frame_in_folder=True, category=None):
    """
    Generate audio -> vision question: listen to sound and choose the matching image.
    
    Args:
        instances (list): List of 4 instances from the same category
        correct_answer (str): Correct answer option ('A', 'B', 'C', or 'D')
        frame_in_folder (bool): Whether images are stored in frame folders
        category (str): Category name for the question
        
    Returns:
        dict: Generated question in structured format
    """
    audio_paths = [os.path.join(root_dir, f"{inst['file']}.wav") for inst in instances]
    
    if frame_in_folder:
        image_frames_folders = [os.path.join(root_dir, f"{inst['file']}_frames") for inst in instances]
        image_paths = []
        for folder in image_frames_folders:
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    image_paths.append(os.path.join(folder, files[0]))
                else:
                    print(f"Warning: No image files found in {folder}")
                    image_paths.append("")  # Empty path for missing images
            else:
                print(f"Warning: Folder not found: {folder}")
                image_paths.append("")  # Empty path for missing folders
    else:
        image_paths = []
        for inst in instances:
            img_path = os.path.join(root_dir, f"{inst['file']}.jpg")
            if os.path.exists(img_path):
                image_paths.append(img_path)
            else:
                print(f"Warning: Image file not found: {img_path}")
                image_paths.append("")
    
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    
    category_text = f" in the '{category}' category" if category else ""
    
    question = {
        "question": f"Which image most likely belongs to the object that makes this sound you hear? Answer with A, B, C, or D",
        "category": category,
        "conditions": {
            "modality": "Audio",
            "input": correct_audio_sample,
        },
        "options": {
            "A": {"modality": "Image", "input": image_paths[0], "class": instances[0]['class']},
            "B": {"modality": "Image", "input": image_paths[1], "class": instances[1]['class']},
            "C": {"modality": "Image", "input": image_paths[2], "class": instances[2]['class']},
            "D": {"modality": "Image", "input": image_paths[3], "class": instances[3]['class']}
        },
        "correct_answer": correct_answer,
        "correct_class": instances[ord(correct_answer) - ord('A')]['class']
    }
    return question


def generate_question_vision_audio(instances, correct_answer, frame_in_folder=True, category=None):
    """
    Generate vision -> audio question: look at image and choose the matching sound.
    
    Args:
        instances (list): List of 4 instances from the same category
        correct_answer (str): Correct answer option ('A', 'B', 'C', or 'D')
        frame_in_folder (bool): Whether images are stored in frame folders
        category (str): Category name for the question
        
    Returns:
        dict: Generated question in structured format
    """
    audio_paths = [os.path.join(root_dir, f"{inst['file']}.wav") for inst in instances]
    correct_idx = ord(correct_answer) - ord('A')
    
    if frame_in_folder:
        image_frames_folder = os.path.join(root_dir, f"{instances[correct_idx]['file']}_frames")
        image_path = os.path.join(image_frames_folder, os.listdir(image_frames_folder)[0]) if os.path.exists(image_frames_folder) else ""
    else:
        image_path = os.path.join(root_dir, f"{instances[correct_idx]['file']}.jpg")

    question = {
        "question": "Look at this image. Which audio clip is most likely produced by what you see? Answer with A, B, C, or D",
        "category": category,
        "conditions": {
            "modality": "Image",
            "input": image_path,
            "class": instances[correct_idx]['class']
        },
        "options": {
            "A": {"modality": "Audio", "input": audio_paths[0], "class": instances[0]['class']},
            "B": {"modality": "Audio", "input": audio_paths[1], "class": instances[1]['class']},
            "C": {"modality": "Audio", "input": audio_paths[2], "class": instances[2]['class']},
            "D": {"modality": "Audio", "input": audio_paths[3], "class": instances[3]['class']}
        },
        "correct_answer": correct_answer,
        "correct_class": instances[correct_idx]['class']
    }
    return question


def generate_question_vision_text(instances, correct_answer, frame_in_folder=True, category=None):
    """
    Generate vision -> text question: look at image and choose the matching description.
    
    Args:
        instances (list): List of 4 instances from the same category
        correct_answer (str): Correct answer option ('A', 'B', 'C', or 'D')
        frame_in_folder (bool): Whether images are stored in frame folders
        category (str): Category name for the question
        
    Returns:
        dict: Generated question in structured format
    """
    correct_idx = ord(correct_answer) - ord('A')
    
    if frame_in_folder:
        image_frames_folder = os.path.join(root_dir, f"{instances[correct_idx]['file']}_frames")
        image_path = os.path.join(image_frames_folder, os.listdir(image_frames_folder)[0]) if os.path.exists(image_frames_folder) else ""
    else:
        image_path = os.path.join(root_dir, f"{instances[correct_idx]['file']}.jpg")
    
    question = {
        "question": "Look at this image. Which text description best matches what you see? Answer with A, B, C, or D",
        "category": category,
        "conditions": {
            "modality": "Image",
            "input": image_path,
            "class": instances[correct_idx]['class']
        },
        "options": {
            "A": {"modality": "Text", "input": instances[0]['class']},
            "B": {"modality": "Text", "input": instances[1]['class']},
            "C": {"modality": "Text", "input": instances[2]['class']},
            "D": {"modality": "Text", "input": instances[3]['class']}
        },
        "correct_answer": correct_answer,
        "correct_class": instances[correct_idx]['class']
    }
    return question


def generate_question_text_vision(instances, correct_answer, frame_in_folder=True, category=None):
    """
    Generate text -> vision question: read description and choose the matching image.
    
    Args:
        instances (list): List of 4 instances from the same category
        correct_answer (str): Correct answer option ('A', 'B', 'C', or 'D')
        frame_in_folder (bool): Whether images are stored in frame folders
        category (str): Category name for the question
        
    Returns:
        dict: Generated question in structured format
    """
    if frame_in_folder:
        image_frames_folders = [os.path.join(root_dir, f"{inst['file']}_frames") for inst in instances]
        image_paths = [os.path.join(folder, os.listdir(folder)[0]) if os.path.exists(folder) else "" for folder in image_frames_folders]
    else:
        image_paths = [os.path.join(root_dir, f"{inst['file']}.jpg") for inst in instances]
    
    correct_idx = ord(correct_answer) - ord('A')
    correct_class = instances[correct_idx]['class']
    
    question = {
        "question": f"Based on this description: '{correct_class}', which image best matches what is described? Answer with A, B, C, or D",
        "category": category,
        "conditions": {
            "modality": "Text",
            "input": correct_class,
        },
        "options": {
            "A": {"modality": "Image", "input": image_paths[0], "class": instances[0]['class']},
            "B": {"modality": "Image", "input": image_paths[1], "class": instances[1]['class']},
            "C": {"modality": "Image", "input": image_paths[2], "class": instances[2]['class']},
            "D": {"modality": "Image", "input": image_paths[3], "class": instances[3]['class']}
        },
        "correct_answer": correct_answer,
        "correct_class": correct_class
    }
    return question


def generate_question_text_audio(instances, correct_answer, category=None):
    """
    Generate text -> audio question: read description and choose the matching sound.
    
    Args:
        instances (list): List of 4 instances from the same category
        correct_answer (str): Correct answer option ('A', 'B', 'C', or 'D')
        category (str): Category name for the question
        
    Returns:
        dict: Generated question in structured format
    """
    audio_paths = [os.path.join(root_dir, f"{inst['file']}.wav") for inst in instances]
    correct_idx = ord(correct_answer) - ord('A')
    correct_class = instances[correct_idx]['class']
    
    question = {
        "question": f"Based on this description: '{correct_class}', which audio clip best matches what is described? Answer with A, B, C, or D",
        "category": category,
        "conditions": {
            "modality": "Text",
            "input": correct_class,
        },
        "options": {
            "A": {"modality": "Audio", "input": audio_paths[0], "class": instances[0]['class']},
            "B": {"modality": "Audio", "input": audio_paths[1], "class": instances[1]['class']},
            "C": {"modality": "Audio", "input": audio_paths[2], "class": instances[2]['class']},
            "D": {"modality": "Audio", "input": audio_paths[3], "class": instances[3]['class']}
        },
        "correct_answer": correct_answer,
        "correct_class": correct_class
    }
    return question


def generate_question_audio_text(instances, correct_answer, category=None):
    """
    Generate audio -> text question: listen to sound and choose the matching description.
    
    Args:
        instances (list): List of 4 instances from the same category
        correct_answer (str): Correct answer option ('A', 'B', 'C', or 'D')
        category (str): Category name for the question
        
    Returns:
        dict: Generated question in structured format
    """
    audio_paths = [os.path.join(root_dir, f"{inst['file']}.wav") for inst in instances]
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    correct_idx = ord(correct_answer) - ord('A')
    
    question = {
        "question": "Listen to this audio clip. Which text description best matches the sound you hear? Answer with A, B, C, or D",
        "category": category,
        "conditions": {
            "modality": "Audio",
            "input": correct_audio_sample,
        },
        "options": {
            "A": {"modality": "Text", "input": instances[0]['class']},
            "B": {"modality": "Text", "input": instances[1]['class']},
            "C": {"modality": "Text", "input": instances[2]['class']},
            "D": {"modality": "Text", "input": instances[3]['class']}
        },
        "correct_answer": correct_answer,
        "correct_class": instances[correct_idx]['class']
    }
    return question

def has_conflicting_classes(classes):
    """
    Check if there are conflicting class combinations that are hard to distinguish visually.
    Currently handles singing vs speaking conflicts for same gender.
    
    Args:
        classes (list): List of class names
        
    Returns:
        bool: True if there are conflicting classes, False otherwise
    """
    # Define conflicting patterns
    conflicting_patterns = [
        # Same gender singing vs speaking
        ('female singing', 'female speech, woman speaking'),
        ('male singing', 'male speech, man speaking'),
        ('child singing', 'child speech, kid speaking'),
        # You can add more conflicting patterns here if needed
    ]
    
    # Check for any conflicting combinations
    for pattern in conflicting_patterns:
        if all(conflict_class in classes for conflict_class in pattern):
            return True
    
    return False


def sample_instances_by_category(category_groups, category, n_samples=4, max_attempts=100):
    """
    Sample instances from a specific category, ensuring all have valid files, different classes,
    and avoiding conflicting class combinations.
    
    Args:
        category_groups (dict): Dictionary of category -> instances mapping
        category (str): Target category to sample from
        n_samples (int): Number of instances to sample (default: 4)
        max_attempts (int): Maximum attempts to find valid instances
        
    Returns:
        list or None: List of sampled instances from different classes, or None if insufficient valid instances
    """
    if category not in category_groups:
        return None
    
    instances = category_groups[category]
    
    # Group instances by class within the category
    class_groups = {}
    for inst in instances:
        class_name = inst['class']
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append(inst)
    
    # Check if we have enough different classes
    if len(class_groups) < n_samples:
        print(f"Warning: Category '{category}' has only {len(class_groups)} classes, need {n_samples}")
        return None
    
    # Try to find valid instances from different classes
    for attempt in range(max_attempts):
        # Randomly select n_samples different classes
        selected_classes = random.sample(list(class_groups.keys()), n_samples)
        
        # Check for conflicting classes
        if has_conflicting_classes(selected_classes):
            continue  # Try again with different classes
        
        # Sample one instance from each selected class
        sampled = []
        all_valid = True
        
        for class_name in selected_classes:
            # Get valid instances from this class
            valid_instances = []
            
            for inst in class_groups[class_name]:
                # Check if audio file exists
                audio_path = os.path.join(root_dir, f"{inst['file']}.wav")
                if not os.path.exists(audio_path):
                    continue
                    
                # Check if image file/folder exists
                if frame_in_folder:
                    frame_folder = os.path.join(root_dir, f"{inst['file']}_frames")
                    if not os.path.exists(frame_folder):
                        continue
                    # Check if folder contains image files
                    image_files = [f for f in os.listdir(frame_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if not image_files:
                        continue
                else:
                    image_path = os.path.join(root_dir, f"{inst['file']}.jpg")
                    if not os.path.exists(image_path):
                        continue
                
                valid_instances.append(inst)
            
            # If no valid instances found for this class, try again
            if not valid_instances:
                all_valid = False
                break
                
            # Randomly select one valid instance from this class
            sampled.append(random.choice(valid_instances))
        
        if all_valid and len(sampled) == n_samples:
            # Verify all classes are different (double-check)
            classes_in_sample = [inst['class'] for inst in sampled]
            if len(set(classes_in_sample)) == n_samples:
                return sampled
    
    print(f"Warning: Could not find {n_samples} valid instances from different classes in category '{category}' after {max_attempts} attempts")
    return None

def get_category_statistics(category_groups):
    """
    Get statistics about classes per category to help with debugging.
    
    Args:
        category_groups (dict): Dictionary of category -> instances mapping
        
    Returns:
        dict: Statistics about each category
    """
    stats = {}
    for category, instances in category_groups.items():
        # Count classes in this category
        classes = set(inst['class'] for inst in instances)
        stats[category] = {
            'total_instances': len(instances),
            'unique_classes': len(classes),
            'classes': list(classes)
        }
    return stats


def generate_all_modality_combinations(instances, correct_answer, frame_in_folder=True, category=None):
    """
    Generate all possible modality combinations for the given instances.
    
    Args:
        instances (list): List of 4 instances from the same category
        correct_answer (str): Correct answer option ('A', 'B', 'C', or 'D')
        frame_in_folder (bool): Whether images are stored in frame folders
        category (str): Category name for the questions
        
    Returns:
        dict: Dictionary containing all generated questions for different modality combinations
    """
    questions = {}
    
    questions['audio_vision'] = generate_question_audio_vision(instances, correct_answer, frame_in_folder, category)
    questions['vision_audio'] = generate_question_vision_audio(instances, correct_answer, frame_in_folder, category)
    questions['vision_text'] = generate_question_vision_text(instances, correct_answer, frame_in_folder, category)
    questions['text_vision'] = generate_question_text_vision(instances, correct_answer, frame_in_folder, category)
    questions['text_audio'] = generate_question_text_audio(instances, correct_answer, category)
    questions['audio_text'] = generate_question_audio_text(instances, correct_answer, category)
    
    return questions


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    
    # Configuration parameters
    DATASET_NAME = 'vggss'
    root_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/vggss_audio_bench"
    frame_in_folder = True  # Set to True if images are in _frames folders
    
    # File paths
    category_file = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/vggss_meta_json/vggss_extend_category.json"
    blacklist_file = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/vggss_meta_json/low_av.txt"
    
    N = 500  # Number of questions to generate per category
    
    # Select modality combinations to generate
    GENERATE_COMBINATIONS = [
        'audio_vision',  # Audio -> Vision
        'vision_audio',  # Vision -> Audio
        'vision_text',   # Vision -> Text
        'text_vision',   # Text -> Vision
        'text_audio',    # Text -> Audio
        'audio_text'     # Audio -> Text
    ]
    
    export_dir = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/Finegrained'
    
    # Load blacklist and category data
    print("Loading blacklist...")
    blacklist = load_blacklist(blacklist_file)
    print(f"Loaded {len(blacklist)} blacklisted instances")
    
    print("Loading category data...")
    category_groups = load_category_data(category_file, blacklist)
    
    print(f"Categories loaded:")
    for category, instances in category_groups.items():
        print(f"  {category}: {len(instances)} instances")
    
    # Get and display class statistics per category
    print("\nAnalyzing class distribution per category...")
    category_stats_detailed = get_category_statistics(category_groups)
    
    print(f"\nDetailed category analysis:")
    for category, stats in category_stats_detailed.items():
        print(f"  {category}: {stats['total_instances']} instances, {stats['unique_classes']} unique classes")
        if stats['unique_classes'] < 4:
            print(f"    WARNING: Only {stats['unique_classes']} classes - cannot generate 4-option questions")
            print(f"    Classes: {', '.join(stats['classes'])}")
    
    # Filter categories with sufficient classes (at least 4 different classes needed)
    valid_categories = {
        cat: instances for cat, instances in category_groups.items() 
        if len(set(inst['class'] for inst in instances)) >= 4
    }
    
    print(f"\nValid categories (>=4 different classes): {len(valid_categories)}")
    excluded_categories = set(category_groups.keys()) - set(valid_categories.keys())
    if excluded_categories:
        print(f"Excluded categories (insufficient classes): {', '.join(excluded_categories)}")
    
    # Initialize question lists for each combination and category statistics
    all_questions = {combo: [] for combo in GENERATE_COMBINATIONS}
    category_stats = {}
    
    # Generate questions for each valid category
    for category in valid_categories:
        print(f"\nGenerating questions for category: {category}")
        category_questions = 0
        
        for i in range(N):
            # Sample 4 instances from current category with different classes
            instances = sample_instances_by_category(category_groups, category, 4)
            if instances is None:
                print(f"  Skipping iteration {i+1} for {category}: insufficient valid instances from different classes")
                continue
            
            # Verify we got different classes (debugging)
            classes_in_sample = [inst['class'] for inst in instances]
            if len(set(classes_in_sample)) != 4:
                print(f"  Warning: Got duplicate classes in sample: {classes_in_sample}")
                continue
            
            correct_answer = random.choice(['A', 'B', 'C', 'D'])
            
            # Generate questions for all modality combinations
            questions = generate_all_modality_combinations(
                instances, correct_answer, 
                frame_in_folder=frame_in_folder, category=category
            )
            
            # Add to corresponding lists
            for combo in GENERATE_COMBINATIONS:
                if combo in questions:
                    all_questions[combo].append(questions[combo])
            
            category_questions += 1
        
        category_stats[category] = category_questions
        print(f"  Generated {category_questions} questions for {category}")
    
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
        "total_categories": len(valid_categories),
        "questions_per_category": N,
        "total_questions_per_combination": sum(len(questions) for questions in all_questions.values()) // len(GENERATE_COMBINATIONS) if GENERATE_COMBINATIONS else 0,
        "category_stats": category_stats,
        "combinations": GENERATE_COMBINATIONS,
        "blacklisted_instances": len(blacklist),
        "excluded_categories": list(excluded_categories),
        "category_class_stats": category_stats_detailed
    }
    
    with open(f"{export_dir}/{DATASET_NAME}_generation_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    print(f"\n=== Generation Summary ===")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Valid categories: {len(valid_categories)}")
    print(f"Excluded categories: {len(excluded_categories)}")
    print(f"Blacklisted instances: {len(blacklist)}")
    print(f"Questions per category: {N}")
    total_questions = sum(len(questions) for questions in all_questions.values()) // len(GENERATE_COMBINATIONS) if GENERATE_COMBINATIONS else 0
    print(f"Total questions per combination: {total_questions}")
    print(f"Generated combinations: {', '.join(GENERATE_COMBINATIONS)}")
    print(f"Export directory: {export_dir}")
    
    print(f"\nCategory breakdown:")
    for category, count in category_stats.items():
        print(f"  {category}: {count} questions")