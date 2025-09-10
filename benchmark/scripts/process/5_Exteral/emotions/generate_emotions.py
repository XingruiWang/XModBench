import os
import re
import random
import json
import librosa
import numpy as np
from pathlib import Path
import soundfile as sf

# ==================== Enhanced Emotion Recognition from Dialog - Multimodal Question Generator ====================


def extract_emotion_from_path(filepath):
    """
    Extract emotion from hierarchical path structure.
    Expected structure: .../category/emotion/files
    Where category is: positive, negative, neutral
    
    Args:
        filepath (str): Full file path
        
    Returns:
        tuple: (category, emotion) or (None, None)
    """
    path_parts = filepath.split(os.sep)
    
    # Look for category and emotion in path
    category = None
    emotion = None
    
    for i, part in enumerate(path_parts):
        part_lower = part.lower()
        if part_lower in ['positive', 'negative', 'neutral']:
            category = part_lower
            # Next part should be the specific emotion
            if i + 1 < len(path_parts):
                emotion = path_parts[i + 1].lower()
            break
    
    return category, emotion


def process_hierarchical_emotion_dataset(face_dataset_dir, audio_dataset_dir):
    """
    Process emotion datasets with hierarchical structure.
    
    Args:
        face_dataset_dir (str): Directory containing face expression videos (mp4)
        audio_dataset_dir (str): Directory containing emotional speech audio (wav)
        
    Returns:
        tuple: (face_emotion_groups, audio_emotion_groups)
    """
    face_emotion_groups = {'positive': {}, 'negative': {}, 'neutral': {}}
    audio_emotion_groups = {'positive': {}, 'negative': {}, 'neutral': {}}
    
    # Process face expression dataset (mp4 files)
    if os.path.exists(face_dataset_dir):
        for root, dirs, files in os.walk(face_dataset_dir):
            for file_name in files:
                if file_name.lower().endswith('.mp4'):
                    video_path = os.path.join(root, file_name)
                    category, emotion = extract_emotion_from_path(video_path)
                    
                    if category and emotion:
                        if emotion not in face_emotion_groups[category]:
                            face_emotion_groups[category][emotion] = []
                        
                        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                            paired_audio_path = video_path.replace('.mp4', '.wav')
                            info = sf.info(paired_audio_path)
                            if info.frames / info.samplerate < 3 or info.frames / info.samplerate > 30:
                                continue
                            face_emotion_groups[category][emotion].append({
                                'name': os.path.splitext(file_name)[0],
                                'emotion': emotion,
                                'category': category,
                                'path': video_path,
                                'type': 'video',
                                'source': f'{category}_{emotion}'
                            })
    
                if file_name.lower().endswith('.wav'):
                    audio_path = os.path.join(root, file_name)
                    category, emotion = extract_emotion_from_path(audio_path)

                    if category and emotion:
                        if emotion not in audio_emotion_groups[category]:
                            audio_emotion_groups[category][emotion] = []
                        
                        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                            # Validate audio file
                            try:
                                
                                info = sf.info(audio_path)
                                if info.frames / info.samplerate < 3 or info.frames / info.samplerate > 30:
                                    continue
                                audio_emotion_groups[category][emotion].append({
                                    'name': os.path.splitext(file_name)[0],
                                    'emotion': emotion,
                                    'category': category,
                                    'path': audio_path,
                                    'type': 'audio',
                                    'duration': info.frames / info.samplerate,
                                    'source': f'{category}_{emotion}'
                                })
                            except Exception as e:
                                print(f"Warning: Cannot validate audio file {audio_path}: {e}")
    
    return face_emotion_groups, audio_emotion_groups


def sample_hierarchical_emotion_instances(face_groups, audio_groups, target_category, target_emotion):
    """
    Sample instances with hierarchical emotion structure.
    - 1 correct answer from target_category/target_emotion
    - 1 neutral option
    - 2 options from opposite category (positive <-> negative)
    
    Args:
        face_groups (dict): Hierarchical face emotion groups
        audio_groups (dict): Hierarchical audio emotion groups
        target_category (str): 'positive' or 'negative' (not 'neutral')
        target_emotion (str): Specific emotion within the category
        
    Returns:
        tuple: (sampled_faces, sampled_audios, correct_idx) or (None, None, None)
    """
    if target_category == 'neutral':
        return None, None, None  # We don't use neutral as target
    
    # Determine opposite category
    opposite_category = 'negative' if target_category == 'positive' else 'positive'
    
    # Check if we have enough data
    if (target_emotion not in face_groups[target_category] or 
        target_emotion not in audio_groups[target_category] or
        len(face_groups['neutral']) == 0 or
        len(audio_groups['neutral']) == 0 or
        len(face_groups[opposite_category]) == 0 or
        len(audio_groups[opposite_category]) == 0):
        return None, None, None
    
    # Sample correct answer (target emotion)
    target_face = random.choice(face_groups[target_category][target_emotion])
    target_audio = random.choice(audio_groups[target_category][target_emotion])
    
    # Sample 1 neutral option
    neutral_emotions = list(face_groups['neutral'].keys())
    if not neutral_emotions:
        return None, None, None
    neutral_emotion = random.choice(neutral_emotions)
    neutral_face = random.choice(face_groups['neutral'][neutral_emotion])
    neutral_audio = random.choice(audio_groups['neutral'][neutral_emotion])
    
    # Sample 2 options from opposite category
    opposite_emotions = list(face_groups[opposite_category].keys())
    if len(opposite_emotions) < 2:
        return None, None, None
    
    selected_opposite_emotions = random.sample(opposite_emotions, 2)
    opposite_faces = []
    opposite_audios = []
    
    
    for opp_emotion in selected_opposite_emotions:
        opp_face = random.choice(face_groups[opposite_category][opp_emotion])
        opp_audio = random.choice(audio_groups[opposite_category][opp_emotion])
        opposite_faces.append(opp_face)
        opposite_audios.append(opp_audio)
    
    # Combine all options: [target, neutral, opposite1, opposite2]
    all_faces = [target_face, neutral_face] + opposite_faces
    all_audios = [target_audio, neutral_audio] + opposite_audios
    
    # Shuffle options but keep track of correct answer
    combined = list(zip(all_faces, all_audios))
    random.shuffle(combined)
    shuffled_faces, shuffled_audios = zip(*combined)
    
    # Find correct answer position
    correct_idx = next(i for i, face in enumerate(shuffled_faces) 
                      if face['category'] == target_category and face['emotion'] == target_emotion)
    correct_answer = chr(ord('A') + correct_idx)
    
    return list(shuffled_faces), list(shuffled_audios), correct_answer


def generate_question_audio_vision(faces, audios, correct_answer, target_category, target_emotion):
    """Generate audio -> vision question: listen to emotional dialog and choose matching facial expression."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_audio = audios[correct_idx]
    
    question = {
        "question": "Listen carefully to this emotional dialog/speech sample. Pay attention to the speaker's tone of voice and dialog to identify the emotion being conveyed through their speech. Then select the video clip that shows the same emotion. Choose A, B, C, or D.",
        "task_description": "Emotion recognition from dialog: Match speech emotion to facial expression",
        "target_emotion": target_emotion,
        "target_category": target_category,
        "conditions": {
            "modality": "Audio",
            "input": correct_audio['path'],
            "emotion": correct_audio['emotion'],
            "category": correct_audio['category'],
            "source": correct_audio['source'],
            "instruction": "Listen to the emotional dialog and identify the speaker's emotional state"
        },
        "options": {
            "A": {"modality": "Video", "input": faces[0]['path'], "emotion": faces[0]['emotion'], "category": faces[0]['category']},
            "B": {"modality": "Video", "input": faces[1]['path'], "emotion": faces[1]['emotion'], "category": faces[1]['category']},
            "C": {"modality": "Video", "input": faces[2]['path'], "emotion": faces[2]['emotion'], "category": faces[2]['category']},
            "D": {"modality": "Video", "input": faces[3]['path'], "emotion": faces[3]['emotion'], "category": faces[3]['category']}
        },
        "correct_answer": correct_answer,
        "correct_emotion": target_emotion,
        "correct_category": target_category
    }
    return question


def generate_question_vision_audio(faces, audios, correct_answer, target_category, target_emotion):
    """Generate vision -> audio question: look at facial expression and choose matching emotional dialog."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_face = faces[correct_idx]
    
    question = {
        "question": "Watch this video clip carefully. Analyze the person's emotional state based on their facial expression and dialog transcript. Then select the dialog/speech audio sample that expresses the same emotion. Choose A, B, C, or D.",
        "task_description": "Emotion recognition from dialog: Match facial expression to speech emotion",
        "target_emotion": target_emotion,
        "target_category": target_category,
        "conditions": {
            "modality": "Video",
            "input": correct_face['path'],
            "emotion": correct_face['emotion'],
            "category": correct_face['category'],
            "source": correct_face['source'],
            "instruction": "Analyze the facial expression to determine the emotional state"
        },
        "options": {
            "A": {"modality": "Audio", "input": audios[0]['path'], "emotion": audios[0]['emotion'], "category": audios[0]['category']},
            "B": {"modality": "Audio", "input": audios[1]['path'], "emotion": audios[1]['emotion'], "category": audios[1]['category']},
            "C": {"modality": "Audio", "input": audios[2]['path'], "emotion": audios[2]['emotion'], "category": audios[2]['category']},
            "D": {"modality": "Audio", "input": audios[3]['path'], "emotion": audios[3]['emotion'], "category": audios[3]['category']}
        },
        "correct_answer": correct_answer,
        "correct_emotion": target_emotion,
        "correct_category": target_category
    }
    return question


def generate_question_vision_text(faces, audios, correct_answer, target_category, target_emotion):
    """Generate vision -> text question: look at facial expression and choose the emotion label."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_face = faces[correct_idx]
    
    # Get emotion labels from all options
    emotion_options = [f"{faces[i]['category'].capitalize()} - {faces[i]['emotion'].capitalize()}" for i in range(4)]
    
    question = {
        "question": "Watch the video clip and the dialog transcript and identify which emotion is being displayed. Choose A, B, C, or D.",
        "target_emotion": target_emotion,
        "target_category": target_category,
        "conditions": {
            "modality": "Video",
            "input": correct_face['path'],
            "emotion": correct_face['emotion'],
            "category": correct_face['category'],
            "source": correct_face['source']
        },
        "options": {
            "A": {"modality": "Text", "input": emotion_options[0]},
            "B": {"modality": "Text", "input": emotion_options[1]},
            "C": {"modality": "Text", "input": emotion_options[2]},
            "D": {"modality": "Text", "input": emotion_options[3]}
        },
        "correct_answer": correct_answer,
        "correct_emotion": target_emotion,
        "correct_category": target_category
    }
    return question


def generate_question_text_vision(faces, audios, correct_answer, target_category, target_emotion):
    """Generate text -> vision question: read emotion label and choose the matching facial expression."""
    question = {
        "question": f"You are given the emotion '{target_category.capitalize()} - {target_emotion.capitalize()}'. Which video clip best demonstrates this emotional state through the facial expression and dialog transcript? Choose A, B, C, or D.",
        "task_description": "Emotion recognition from dialog: Match emotion label to facial expression",
        "target_emotion": target_emotion,
        "target_category": target_category,
        "conditions": {
            "modality": "Text",
            "input": f"{target_category.capitalize()} - {target_emotion.capitalize()}",
            "instruction": f"Find the facial expression that matches the emotion: {target_category.capitalize()} - {target_emotion.capitalize()}"
        },
        "options": {
            "A": {"modality": "Video", "input": faces[0]['path'], "emotion": faces[0]['emotion'], "category": faces[0]['category']},
            "B": {"modality": "Video", "input": faces[1]['path'], "emotion": faces[1]['emotion'], "category": faces[1]['category']},
            "C": {"modality": "Video", "input": faces[2]['path'], "emotion": faces[2]['emotion'], "category": faces[2]['category']},
            "D": {"modality": "Video", "input": faces[3]['path'], "emotion": faces[3]['emotion'], "category": faces[3]['category']}
        },
        "correct_answer": correct_answer,
        "correct_emotion": target_emotion,
        "correct_category": target_category
    }
    return question


def generate_question_text_audio(faces, audios, correct_answer, target_category, target_emotion):
    """Generate text -> audio question: read emotion label and choose the matching emotional dialog."""
    question = {
        "question": f"You are given the emotion '{target_category.capitalize()} - {target_emotion.capitalize()}'. Which dialog/speech audio sample best demonstrates this emotion through the speaker's voice and delivery? Choose A, B, C, or D.",
        "task_description": "Emotion recognition from dialog: Match emotion label to speech emotion",
        "target_emotion": target_emotion,
        "target_category": target_category,
        "conditions": {
            "modality": "Text",
            "input": f"{target_category.capitalize()} - {target_emotion.capitalize()}",
            "instruction": f"Find the speech sample that expresses the emotion: {target_category.capitalize()} - {target_emotion.capitalize()}"
        },
        "options": {
            "A": {"modality": "Audio", "input": audios[0]['path'], "emotion": audios[0]['emotion'], "category": audios[0]['category']},
            "B": {"modality": "Audio", "input": audios[1]['path'], "emotion": audios[1]['emotion'], "category": audios[1]['category']},
            "C": {"modality": "Audio", "input": audios[2]['path'], "emotion": audios[2]['emotion'], "category": audios[2]['category']},
            "D": {"modality": "Audio", "input": audios[3]['path'], "emotion": audios[3]['emotion'], "category": audios[3]['category']}
        },
        "correct_answer": correct_answer,
        "correct_emotion": target_emotion,
        "correct_category": target_category
    }
    return question


def generate_question_audio_text(faces, audios, correct_answer, target_category, target_emotion):
    """Generate audio -> text question: listen to emotional dialog and choose the emotion label."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_audio = audios[correct_idx]
    
    # Get emotion labels from all options
    emotion_options = [f"{audios[i]['category'].capitalize()} - {audios[i]['emotion'].capitalize()}" for i in range(4)]
    
    question = {
        "question": "Listen to the dialog audio clip and identify which emotion is being expressed. Choose A, B, C, or D.",
        "task_description": "Emotion recognition from dialog: Identify emotion from speech patterns",
        "target_emotion": target_emotion,
        "target_category": target_category,
        "conditions": {
            "modality": "Audio",
            "input": correct_audio['path'],
            "emotion": correct_audio['emotion'],
            "category": correct_audio['category'],
            "source": correct_audio['source'],
            "instruction": "Listen to the dialog and identify the emotion being expressed through vocal patterns"
        },
        "options": {
            "A": {"modality": "Text", "input": emotion_options[0]},
            "B": {"modality": "Text", "input": emotion_options[1]},
            "C": {"modality": "Text", "input": emotion_options[2]},
            "D": {"modality": "Text", "input": emotion_options[3]}
        },
        "correct_answer": correct_answer,
        "correct_emotion": target_emotion,
        "correct_category": target_category
    }
    return question


def generate_all_modality_combinations(faces, audios, correct_answer, target_category, target_emotion):
    """Generate all possible modality combinations for emotion recognition from dialog."""
    questions = {}
    
    questions['audio_vision'] = generate_question_audio_vision(faces, audios, correct_answer, target_category, target_emotion)
    questions['vision_audio'] = generate_question_vision_audio(faces, audios, correct_answer, target_category, target_emotion)
    questions['vision_text'] = generate_question_vision_text(faces, audios, correct_answer, target_category, target_emotion)
    questions['text_vision'] = generate_question_text_vision(faces, audios, correct_answer, target_category, target_emotion)
    questions['text_audio'] = generate_question_text_audio(faces, audios, correct_answer, target_category, target_emotion)
    questions['audio_text'] = generate_question_audio_text(faces, audios, correct_answer, target_category, target_emotion)
    
    return questions


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    
    # Configuration parameters
    DATASET_NAME = 'emotion_classification'
    face_dataset_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/emotions/processed_emotion_data"  # Update to your face dataset path (mp4 files)
    audio_dataset_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/emotions/processed_emotion_data"  # Update to your audio dataset path (wav files)
    
    N = 100  # Number of questions to generate per emotion (as requested)
    
    # Select modality combinations to generate
    GENERATE_COMBINATIONS = [
        'audio_vision',  # Dialog -> Face (Emotional speech -> Facial expression)
        'vision_audio',  # Face -> Dialog (Facial expression -> Emotional speech)
        'vision_text',   # Face -> Text (Facial expression -> Emotion label)
        'text_vision',   # Text -> Face (Emotion label -> Facial expression)
        'text_audio',    # Text -> Dialog (Emotion label -> Emotional speech)
        'audio_text'     # Dialog -> Text (Emotional speech -> Emotion label)
    ]
    
    export_dir = '/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/05_Exteral/emotion_classification'
    
    # Process the emotion datasets
    print("Processing hierarchical emotion datasets for dialog-based emotion recognition...")
    face_emotion_groups, audio_emotion_groups = process_hierarchical_emotion_dataset(face_dataset_dir, audio_dataset_dir)
    
    print(f"Face emotion groups loaded:")
    for category in ['positive', 'negative', 'neutral']:
        print(f"  {category.capitalize()}:")
        for emotion, instances in face_emotion_groups[category].items():
            print(f"    {emotion}: {len(instances)} video files")
    
    print(f"Audio emotion groups loaded:")
    for category in ['positive', 'negative', 'neutral']:
        print(f"  {category.capitalize()}:")
        for emotion, instances in audio_emotion_groups[category].items():
            print(f"    {emotion}: {len(instances)} audio files")
    
    # Find valid target emotions (only positive and negative, with sufficient data)
    valid_target_emotions = []
    for category in ['positive', 'negative']:
        for emotion in face_emotion_groups[category].keys():
            if (emotion in audio_emotion_groups[category] and
                len(face_emotion_groups[category][emotion]) >= 1 and
                len(audio_emotion_groups[category][emotion]) >= 1):
                valid_target_emotions.append((category, emotion))
    
    print(f"\nValid target emotions: {len(valid_target_emotions)}")
    for category, emotion in valid_target_emotions:
        print(f"  {category} - {emotion}")
    
    if len(valid_target_emotions) == 0:
        print("Error: No valid target emotions found!")
        exit(1)
    
    # Initialize question lists for each combination
    all_questions = {combo: [] for combo in GENERATE_COMBINATIONS}
    emotion_stats = {}
    
    # Generate questions for each valid target emotion
    for target_category, target_emotion in valid_target_emotions:
        print(f"\nGenerating dialog emotion recognition questions with '{target_category} - {target_emotion}' as target...")
        emotion_questions = 0
        
        for i in range(N):
            # Sample hierarchical instances
            sampled_faces, sampled_audios, correct_answer = sample_hierarchical_emotion_instances(
                face_emotion_groups, audio_emotion_groups, target_category, target_emotion
            )
            
            if sampled_faces is None or sampled_audios is None:
                print(f"  Skipping iteration {i+1} for {target_category}-{target_emotion}: sampling failed")
                continue
            
            # Generate questions for all modality combinations
            questions = generate_all_modality_combinations(
                sampled_faces, sampled_audios, correct_answer, target_category, target_emotion
            )
            
            # Add to corresponding lists
            for combo in GENERATE_COMBINATIONS:
                if combo in questions:
                    all_questions[combo].append(questions[combo])
            
            emotion_questions += 1
        
        emotion_key = f"{target_category}_{target_emotion}"
        emotion_stats[emotion_key] = emotion_questions
        print(f"  Generated {emotion_questions} questions with {target_category}-{target_emotion} as target")
    
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
        "task_type": "dialog_emotion_recognition_multimodal",
        "total_target_emotions": len(valid_target_emotions),
        "valid_target_emotions": [f"{cat}_{emo}" for cat, emo in valid_target_emotions],
        "questions_per_emotion": N,
        "total_questions_per_combination": sum(len(questions) for questions in all_questions.values()) // len(GENERATE_COMBINATIONS),
        "emotion_stats": emotion_stats,
        "combinations": GENERATE_COMBINATIONS,
        "sampling_strategy": "1_correct + 1_neutral + 2_opposite_category",
        "face_media_type": "mp4_videos",
        "audio_media_type": "wav_dialog_files",
        "task_description": "Cross-modal emotion recognition from dialog and conversation data"
    }
    
    with open(f"{export_dir}/{DATASET_NAME}_generation_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    print(f"\n=== Dialog Emotion Recognition Generation Summary ===")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Task: Cross-modal emotion recognition from dialog/conversation")
    print(f"Valid target emotions: {len(valid_target_emotions)}")
    print(f"Sampling strategy: 1 correct + 1 neutral + 2 from opposite category")
    print(f"Questions per target emotion: {N}")
    print(f"Face media: MP4 videos")
    print(f"Audio media: WAV dialog/speech files")
    print(f"Total questions per combination: {sum(len(questions) for questions in all_questions.values()) // len(GENERATE_COMBINATIONS)}")
    print(f"Generated combinations: {', '.join(GENERATE_COMBINATIONS)}")
    print(f"Export directory: {export_dir}")
    
    print(f"\nTarget emotion breakdown:")
    for emotion_key, count in emotion_stats.items():
        print(f"  {emotion_key}: {count} questions")