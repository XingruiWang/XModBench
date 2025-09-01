import os
import re
import random
import json
import librosa
import numpy as np
from pathlib import Path


# ==================== Emotion Classification Multimodal Question-Answer Generator ====================


def extract_emotion_from_filename(filename_or_path):
    """
    Extract emotion from filename/path.
    
    Args:
        filename_or_path (str): File name or path containing emotion info
        
    Returns:
        str: Emotion name (angry, happy, sad, etc.) or None
    """
    # Extract base filename
    filename = os.path.basename(filename_or_path)
    
    # Define emotion mappings
    emotion_mappings = {
        'angry': ['angry', 'anger'],
        'disgust': ['disgust'],
        'fear': ['fear'],
        'happy': ['happy', 'happiness'],
        'neutral': ['neutral'],
        'sad': ['sad', 'sadness'],
        # 'surprise': ['surprise', 'pleasant_surprise']
    }
    
    # Check for emotion patterns in filename
    filename_lower = filename.lower()
    
    # Handle OAF/YAF patterns (OAF_angry, YAF_happy, etc.)
    if 'oaf_' in filename_lower or 'yaf_' in filename_lower:
        for emotion, variants in emotion_mappings.items():
            for variant in variants:
                if variant in filename_lower:
                    return emotion
    
    # Handle direct emotion folder names
    for emotion, variants in emotion_mappings.items():
        for variant in variants:
            if variant in filename_lower:
                return emotion
    
    return None


def process_emotion_dataset(face_dataset_dir, audio_dataset_dir):
    """
    Process emotion datasets and organize by emotions.
    
    Args:
        face_dataset_dir (str): Directory containing face expression images
        audio_dataset_dir (str): Directory containing emotional speech audio
        
    Returns:
        tuple: (face_emotion_groups, audio_emotion_groups)
    """
    face_emotion_groups = {}
    audio_emotion_groups = {}
    
    # Process face expression dataset (validation set)
    validation_dir = os.path.join(face_dataset_dir, "validation")
    if os.path.exists(validation_dir):
        for emotion_folder in os.listdir(validation_dir):
            emotion_path = os.path.join(validation_dir, emotion_folder)
            if os.path.isdir(emotion_path):
                # Use folder name as emotion
                emotion = emotion_folder.lower()
                face_emotion_groups[emotion] = []
                
                # Get all image files in emotion folder
                for img_file in os.listdir(emotion_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(emotion_path, img_file)
                        if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                            face_emotion_groups[emotion].append({
                                'name': os.path.splitext(img_file)[0],
                                'emotion': emotion,
                                'path': img_path,
                                'type': 'face',
                                'source': 'face_expression_dataset'
                            })
    
    # Process TESS emotional speech dataset
    if os.path.exists(audio_dataset_dir):
        for audio_folder in os.listdir(audio_dataset_dir):
            audio_folder_path = os.path.join(audio_dataset_dir, audio_folder)
            if os.path.isdir(audio_folder_path):
                # Extract emotion from folder name (e.g., OAF_angry -> angry)
                emotion = extract_emotion_from_filename(audio_folder)
                if emotion:
                    if emotion not in audio_emotion_groups:
                        audio_emotion_groups[emotion] = []
                    
                    # Get all audio files in the folder
                    for audio_file in os.listdir(audio_folder_path):
                        if audio_file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                            audio_path = os.path.join(audio_folder_path, audio_file)
                            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                                # Validate audio file
                                try:
                                    import soundfile as sf
                                    info = sf.info(audio_path)
                                    if info.frames > 0:
                                        audio_emotion_groups[emotion].append({
                                            'name': os.path.splitext(audio_file)[0],
                                            'emotion': emotion,
                                            'path': audio_path,
                                            'type': 'audio',
                                            'duration': info.frames / info.samplerate,
                                            'source': audio_folder
                                        })
                                except Exception as e:
                                    print(f"Warning: Cannot validate audio file {audio_path}: {e}")
    
    return face_emotion_groups, audio_emotion_groups


def sample_mixed_emotion_instances(face_groups, audio_groups, target_emotion, n_samples=4):
    """
    Sample instances from different emotions, ensuring one correct answer from target emotion.
    
    Args:
        face_groups (dict): Dictionary of emotion -> face instances
        audio_groups (dict): Dictionary of emotion -> audio instances  
        target_emotion (str): The correct emotion for this question
        n_samples (int): Total number of options (default: 4)
        
    Returns:
        tuple: (sampled_faces, sampled_audios, correct_idx) or (None, None, None) if sampling fails
    """
    all_emotions = list(set(face_groups.keys()) & set(audio_groups.keys()))
    if target_emotion not in all_emotions or len(all_emotions) < n_samples:
        return None, None, None
    
    # Sample other emotions (distractors)
    other_emotions = [e for e in all_emotions if e != target_emotion]
    if len(other_emotions) < n_samples - 1:
        return None, None, None
    
    distractor_emotions = random.sample(other_emotions, n_samples - 1)
    selected_emotions = [target_emotion] + distractor_emotions
    random.shuffle(selected_emotions)
    
    # Find the correct answer position
    correct_idx = selected_emotions.index(target_emotion)
    correct_answer = chr(ord('A') + correct_idx)
    
    # Sample one instance from each selected emotion
    sampled_faces = []
    sampled_audios = []
    
    # Sample audio
    audio_person = ''
    audio_word = ''
    for emotion in selected_emotions:
        # Sample face
        if emotion in face_groups and len(face_groups[emotion]) > 0:
            face = random.choice(face_groups[emotion])
            sampled_faces.append(face)
        else:
            return None, None, None
            

        if emotion in audio_groups and len(audio_groups[emotion]) > 0:
            audio = random.choice(audio_groups[emotion])
            
            if audio_person == '' and audio_word == '':
                audio_person, audio_word, _ = audio['name'].split('_')
            else:
                matched_sample_audio = [a for a in audio_groups[emotion] if a['name'].split('_')[0] == audio_person and a['name'].split('_')[1] == audio_word]
                if len(matched_sample_audio) > 0:
                    audio = random.choice(matched_sample_audio)
                else:
                    return None, None, None
            sampled_audios.append(audio)
        else:
            return None, None, None
    return sampled_faces, sampled_audios, correct_answer


def generate_question_audio_vision(faces, audios, correct_answer, target_emotion):
    """Generate audio -> vision question: listen to emotional speech and choose matching facial expression."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_audio = audios[correct_idx]
    
    question = {
        "question": "Listen to this emotional speech. Which facial expression shows the same emotion? Answer with A, B, C, or D",
        "target_emotion": target_emotion,
        "conditions": {
            "modality": "Audio",
            "input": correct_audio['path'],
            "emotion": correct_audio['emotion'],
            "source": correct_audio['source']
        },
        "options": {
            "A": {"modality": "Image", "input": faces[0]['path'], "emotion": faces[0]['emotion']},
            "B": {"modality": "Image", "input": faces[1]['path'], "emotion": faces[1]['emotion']},
            "C": {"modality": "Image", "input": faces[2]['path'], "emotion": faces[2]['emotion']},
            "D": {"modality": "Image", "input": faces[3]['path'], "emotion": faces[3]['emotion']}
        },
        "correct_answer": correct_answer,
        "correct_emotion": target_emotion
    }
    return question


def generate_question_vision_audio(faces, audios, correct_answer, target_emotion):
    """Generate vision -> audio question: look at facial expression and choose matching emotional speech."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_face = faces[correct_idx]
    
    question = {
        "question": "Look at this facial expression. Which speech sample expresses the same emotion? Answer with A, B, C, or D",
        "target_emotion": target_emotion,
        "conditions": {
            "modality": "Image",
            "input": correct_face['path'],
            "emotion": correct_face['emotion'],
            "source": correct_face['source']
        },
        "options": {
            "A": {"modality": "Audio", "input": audios[0]['path'], "emotion": audios[0]['emotion']},
            "B": {"modality": "Audio", "input": audios[1]['path'], "emotion": audios[1]['emotion']},
            "C": {"modality": "Audio", "input": audios[2]['path'], "emotion": audios[2]['emotion']},
            "D": {"modality": "Audio", "input": audios[3]['path'], "emotion": audios[3]['emotion']}
        },
        "correct_answer": correct_answer,
        "correct_emotion": target_emotion
    }
    return question


def generate_question_vision_text(faces, audios, correct_answer, target_emotion):
    """Generate vision -> text question: look at facial expression and choose the emotion label."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_face = faces[correct_idx]
    
    # Get emotion names from all options
    emotion_options = [faces[i]['emotion'] for i in range(4)]
    
    question = {
        "question": "Look at this facial expression. Which emotion does this face show? Answer with A, B, C, or D",
        "target_emotion": target_emotion,
        "conditions": {
            "modality": "Image",
            "input": correct_face['path'],
            "emotion": correct_face['emotion'],
            "source": correct_face['source']
        },
        "options": {
            "A": {"modality": "Text", "input": emotion_options[0].capitalize()},
            "B": {"modality": "Text", "input": emotion_options[1].capitalize()},
            "C": {"modality": "Text", "input": emotion_options[2].capitalize()},
            "D": {"modality": "Text", "input": emotion_options[3].capitalize()}
        },
        "correct_answer": correct_answer,
        "correct_emotion": target_emotion
    }
    return question


def generate_question_text_vision(faces, audios, correct_answer, target_emotion):
    """Generate text -> vision question: read emotion label and choose the matching facial expression."""
    correct_idx = ord(correct_answer) - ord('A')
    
    question = {
        "question": f"Based on the emotion '{target_emotion.capitalize()}', which facial expression best represents this emotion? Answer with A, B, C, or D",
        "target_emotion": target_emotion,
        "conditions": {
            "modality": "Text",
            "input": target_emotion.capitalize(),
        },
        "options": {
            "A": {"modality": "Image", "input": faces[0]['path'], "emotion": faces[0]['emotion']},
            "B": {"modality": "Image", "input": faces[1]['path'], "emotion": faces[1]['emotion']},
            "C": {"modality": "Image", "input": faces[2]['path'], "emotion": faces[2]['emotion']},
            "D": {"modality": "Image", "input": faces[3]['path'], "emotion": faces[3]['emotion']}
        },
        "correct_answer": correct_answer,
        "correct_emotion": target_emotion
    }
    return question


def generate_question_text_audio(faces, audios, correct_answer, target_emotion):
    """Generate text -> audio question: read emotion label and choose the matching emotional speech."""
    correct_idx = ord(correct_answer) - ord('A')
    
    question = {
        "question": f"Based on the emotion '{target_emotion.capitalize()}', which speech sample best expresses this emotion? Answer with A, B, C, or D",
        "target_emotion": target_emotion,
        "conditions": {
            "modality": "Text",
            "input": target_emotion.capitalize(),
        },
        "options": {
            "A": {"modality": "Audio", "input": audios[0]['path'], "emotion": audios[0]['emotion']},
            "B": {"modality": "Audio", "input": audios[1]['path'], "emotion": audios[1]['emotion']},
            "C": {"modality": "Audio", "input": audios[2]['path'], "emotion": audios[2]['emotion']},
            "D": {"modality": "Audio", "input": audios[3]['path'], "emotion": audios[3]['emotion']}
        },
        "correct_answer": correct_answer,
        "correct_emotion": target_emotion
    }
    return question


def generate_question_audio_text(faces, audios, correct_answer, target_emotion):
    """Generate audio -> text question: listen to emotional speech and choose the emotion label."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_audio = audios[correct_idx]
    
    # Get emotion names from all options
    emotion_options = [audios[i]['emotion'] for i in range(4)]
    
    question = {
        "question": "Listen to this emotional speech. Which emotion is being expressed? Answer with A, B, C, or D",
        "target_emotion": target_emotion,
        "conditions": {
            "modality": "Audio",
            "input": correct_audio['path'],
            "emotion": correct_audio['emotion'],
            "source": correct_audio['source']
        },
        "options": {
            "A": {"modality": "Text", "input": emotion_options[0].capitalize()},
            "B": {"modality": "Text", "input": emotion_options[1].capitalize()},
            "C": {"modality": "Text", "input": emotion_options[2].capitalize()},
            "D": {"modality": "Text", "input": emotion_options[3].capitalize()}
        },
        "correct_answer": correct_answer,
        "correct_emotion": target_emotion
    }
    return question


def generate_all_modality_combinations(faces, audios, correct_answer, target_emotion):
    """Generate all possible modality combinations for emotion classification."""
    questions = {}
    
    questions['audio_vision'] = generate_question_audio_vision(faces, audios, correct_answer, target_emotion)
    questions['vision_audio'] = generate_question_vision_audio(faces, audios, correct_answer, target_emotion)
    questions['vision_text'] = generate_question_vision_text(faces, audios, correct_answer, target_emotion)
    questions['text_vision'] = generate_question_text_vision(faces, audios, correct_answer, target_emotion)
    questions['text_audio'] = generate_question_text_audio(faces, audios, correct_answer, target_emotion)
    questions['audio_text'] = generate_question_audio_text(faces, audios, correct_answer, target_emotion)
    
    return questions


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    
    # Configuration parameters
    DATASET_NAME = 'emotion_classification'
    face_dataset_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/emotions/face-expression-recognition-dataset"  # Update to your face dataset path
    audio_dataset_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/emotions/TESS_Toronto_emotional_speech_set_data"  # Update to your audio dataset path
    
    N = 100  # Number of questions to generate per emotion
    
    # Select modality combinations to generate
    GENERATE_COMBINATIONS = [
        'audio_vision',  # Speech -> Face (Emotional speech -> Facial expression)
        'vision_audio',  # Face -> Speech (Facial expression -> Emotional speech)
        'vision_text',   # Face -> Text (Facial expression -> Emotion label)
        'text_vision',   # Text -> Face (Emotion label -> Facial expression)
        'text_audio',    # Text -> Speech (Emotion label -> Emotional speech)
        'audio_text'     # Speech -> Text (Emotional speech -> Emotion label)
    ]
    
    export_dir = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/05_Exteral/emotion_classification'
    
    # Process the emotion datasets
    print("Processing emotion datasets...")
    face_emotion_groups, audio_emotion_groups = process_emotion_dataset(face_dataset_dir, audio_dataset_dir)
    
    print(f"Face emotion groups loaded:")
    for emotion, instances in face_emotion_groups.items():
        print(f"  {emotion}: {len(instances)} face images")
    
    print(f"Audio emotion groups loaded:")
    for emotion, instances in audio_emotion_groups.items():
        print(f"  {emotion}: {len(instances)} audio clips")
    
    # Find common emotions between face and audio datasets
    common_emotions = set(face_emotion_groups.keys()) & set(audio_emotion_groups.keys())
    valid_emotions = [emotion for emotion in common_emotions 
                     if len(face_emotion_groups[emotion]) >= 1 and len(audio_emotion_groups[emotion]) >= 1]
    
    print(f"\nValid emotions (have both face and audio data): {len(valid_emotions)}")
    print(f"Valid emotions: {valid_emotions}")
    
    if len(valid_emotions) < 4:
        print("Error: Need at least 4 emotions to generate questions!")
        exit(1)
    
    # Initialize question lists for each combination and emotion statistics
    all_questions = {combo: [] for combo in GENERATE_COMBINATIONS}
    emotion_stats = {}
    
    # Generate questions for each valid emotion as the target (correct) emotion
    for target_emotion in valid_emotions:
        print(f"\nGenerating questions with '{target_emotion}' as target emotion...")
        emotion_questions = 0
        
        for i in range(N):
            # Sample mixed instances (1 correct + 3 distractors)
            sampled_faces, sampled_audios, correct_answer = sample_mixed_emotion_instances(
                face_emotion_groups, audio_emotion_groups, target_emotion, 4
            )
            
            if sampled_faces is None or sampled_audios is None:
                print(f"  Skipping iteration {i+1} for {target_emotion}: sampling failed")
                continue
            
            # Generate questions for all modality combinations
            questions = generate_all_modality_combinations(
                sampled_faces, sampled_audios, correct_answer, target_emotion
            )
            
            # Add to corresponding lists
            for combo in GENERATE_COMBINATIONS:
                if combo in questions:
                    all_questions[combo].append(questions[combo])
            
            emotion_questions += 1
        
        emotion_stats[target_emotion] = emotion_questions
        print(f"  Generated {emotion_questions} questions with {target_emotion} as target")
    
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
        "total_emotions": len(valid_emotions),
        "valid_emotions": valid_emotions,
        "questions_per_emotion": N,
        "total_questions_per_combination": sum(len(questions) for questions in all_questions.values()) // len(GENERATE_COMBINATIONS),
        "emotion_stats": emotion_stats,
        "combinations": GENERATE_COMBINATIONS,
        "task_type": "emotion_matching_across_modalities",
        "face_dataset_source": "face-expression-recognition-dataset/validation",
        "audio_dataset_source": "TESS_Toronto_emotional_speech_set_data"
    }
    
    with open(f"{export_dir}/{DATASET_NAME}_generation_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    print(f"\n=== Generation Summary ===")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Valid emotions: {len(valid_emotions)}")
    print(f"Task: Emotion matching across modalities")
    print(f"Questions per target emotion: {N}")
    print(f"Face dataset: face-expression-recognition-dataset/validation")
    print(f"Audio dataset: TESS Toronto emotional speech set")
    print(f"Total questions per combination: {sum(len(questions) for questions in all_questions.values()) // len(GENERATE_COMBINATIONS)}")
    print(f"Generated combinations: {', '.join(GENERATE_COMBINATIONS)}")
    print(f"Export directory: {export_dir}")
    
    print(f"\nTarget emotion breakdown:")
    for emotion, count in emotion_stats.items():
        print(f"  {emotion}: {count} questions")