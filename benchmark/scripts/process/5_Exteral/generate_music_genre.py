import os
import re
import random
import json
import librosa
import numpy as np
from pathlib import Path


# ==================== Music Genre Multimodal Question-Answer Generator ====================


def extract_high_quality_audio_segment(audio_path, target_duration=10, sr=22050):
    """
    Extract a high-quality 10-second segment from a 30-second audio file.
    
    Args:
        audio_path (str): Path to the audio file
        target_duration (int): Target duration in seconds (default: 10)
        sr (int): Sample rate (default: 22050)
        
    Returns:
        tuple: (audio_segment, start_time) or (None, None) if extraction fails
    """
    try:
        # Load the full audio file
        y, sr = librosa.load(audio_path, sr=sr)
        total_duration = len(y) / sr
        
        if total_duration < target_duration:
            print(f"Warning: Audio {audio_path} is shorter than {target_duration} seconds")
            return y, 0
        
        # Calculate energy for different segments to find the most active part
        hop_length = 512
        frame_length = 2048
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Convert frame indices to time
        times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
        
        # Find the segment with highest average energy
        segment_length_frames = int(target_duration * sr / hop_length)
        best_start_frame = 0
        best_energy = 0
        
        for i in range(len(rms) - segment_length_frames):
            segment_energy = np.mean(rms[i:i + segment_length_frames])
            if segment_energy > best_energy:
                best_energy = segment_energy
                best_start_frame = i
        
        # Convert back to audio samples
        start_sample = int(best_start_frame * hop_length)
        end_sample = start_sample + int(target_duration * sr)
        
        audio_segment = y[start_sample:end_sample]
        start_time = start_sample / sr
        
        return audio_segment, start_time
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None


def process_music_dataset(root_dir):
    """
    Process the music dataset and organize by genres separately for covers and audio.
    
    Args:
        root_dir (str): Root directory of the music dataset
        
    Returns:
        tuple: (cover_groups, audio_groups) - separate dictionaries for covers and audio by genre
    """
    cover_groups = {}
    audio_groups = {}
    
    # Process covers (album cover images)
    covers_dir = os.path.join(root_dir, "covers")
    if os.path.exists(covers_dir):
        for genre in os.listdir(covers_dir):
            genre_path = os.path.join(covers_dir, genre)
            if os.path.isdir(genre_path):
                cover_groups[genre] = []
                
                # Get all image files in the genre directory
                for img_file in os.listdir(genre_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        cover_groups[genre].append({
                            'name': os.path.splitext(img_file)[0],
                            'genre': genre,
                            'path': os.path.join(genre_path, img_file),
                            'type': 'cover'
                        })
    
    # Process original genre audio files
    genres_original_dir = os.path.join(root_dir, "genres_original")
    if os.path.exists(genres_original_dir):
        for genre in os.listdir(genres_original_dir):
            genre_path = os.path.join(genres_original_dir, genre)
            if os.path.isdir(genre_path):
                audio_groups[genre] = []
                
                # Get all audio files in the genre directory
                for audio_file in os.listdir(genre_path):
                    if audio_file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                        audio_groups[genre].append({
                            'name': os.path.splitext(audio_file)[0],
                            'genre': genre,
                            'path': os.path.join(genre_path, audio_file),
                            'type': 'audio'
                        })
    
    return cover_groups, audio_groups


def sample_mixed_instances(cover_groups, audio_groups, target_genre, n_samples=4):
    """
    Sample instances from different genres, ensuring one correct answer from target genre.
    
    Args:
        cover_groups (dict): Dictionary of genre -> cover instances
        audio_groups (dict): Dictionary of genre -> audio instances  
        target_genre (str): The correct genre for this question
        n_samples (int): Total number of options (default: 4)
        
    Returns:
        tuple: (sampled_covers, sampled_audios, correct_idx) or (None, None, None) if sampling fails
    """
    all_genres = list(set(cover_groups.keys()) & set(audio_groups.keys()))
    if target_genre not in all_genres or len(all_genres) < n_samples:
        return None, None, None
    
    # Sample other genres (distractors)
    other_genres = [g for g in all_genres if g != target_genre]
    if len(other_genres) < n_samples - 1:
        return None, None, None
    
    distractor_genres = random.sample(other_genres, n_samples - 1)
    selected_genres = [target_genre] + distractor_genres
    random.shuffle(selected_genres)
    
    # Find the correct answer position
    correct_idx = selected_genres.index(target_genre)
    correct_answer = chr(ord('A') + correct_idx)
    
    # Sample one instance from each selected genre
    sampled_covers = []
    sampled_audios = []
    
    for genre in selected_genres:
        # Sample cover
        if genre in cover_groups and len(cover_groups[genre]) > 0:
            cover = random.choice(cover_groups[genre])
            sampled_covers.append(cover)
        else:
            return None, None, None
            
        # Sample audio
        if genre in audio_groups and len(audio_groups[genre]) > 0:
            audio = random.choice(audio_groups[genre])
            sampled_audios.append(audio)
        else:
            return None, None, None
    
    return sampled_covers, sampled_audios, correct_answer


def generate_question_audio_vision(covers, audios, correct_answer, target_genre):
    """Generate audio -> vision question: listen to music and choose the matching genre cover."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_audio = audios[correct_idx]
    
    question = {
        "question": "Listen to this music clip. Which album cover is from the same music genre? Answer with A, B, C, or D",
        "target_genre": target_genre,
        "conditions": {
            "modality": "Audio",
            "input": correct_audio['path'],
            "genre": correct_audio['genre']
        },
        "options": {
            "A": {"modality": "Image", "input": covers[0]['path'], "genre": covers[0]['genre']},
            "B": {"modality": "Image", "input": covers[1]['path'], "genre": covers[1]['genre']},
            "C": {"modality": "Image", "input": covers[2]['path'], "genre": covers[2]['genre']},
            "D": {"modality": "Image", "input": covers[3]['path'], "genre": covers[3]['genre']}
        },
        "correct_answer": correct_answer,
        "correct_genre": target_genre
    }
    return question


def generate_question_vision_audio(covers, audios, correct_answer, target_genre):
    """Generate vision -> audio question: look at album cover and choose the matching genre music."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_cover = covers[correct_idx]
    
    question = {
        "question": "Look at this album cover. Which music clip is from the same genre as this cover suggests? Answer with A, B, C, or D",
        "target_genre": target_genre,
        "conditions": {
            "modality": "Image",
            "input": correct_cover['path'],
            "genre": correct_cover['genre']
        },
        "options": {
            "A": {"modality": "Audio", "input": audios[0]['path'], "genre": audios[0]['genre']},
            "B": {"modality": "Audio", "input": audios[1]['path'], "genre": audios[1]['genre']},
            "C": {"modality": "Audio", "input": audios[2]['path'], "genre": audios[2]['genre']},
            "D": {"modality": "Audio", "input": audios[3]['path'], "genre": audios[3]['genre']}
        },
        "correct_answer": correct_answer,
        "correct_genre": target_genre
    }
    return question


def generate_question_vision_text(covers, audios, correct_answer, target_genre):
    """Generate vision -> text question: look at album cover and choose the matching genre name."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_cover = covers[correct_idx]
    
    # Get genre names from all options
    genre_options = [covers[i]['genre'] for i in range(4)]
    
    question = {
        "question": "Look at this album cover. Which music genre does this cover art represent? Answer with A, B, C, or D",
        "target_genre": target_genre,
        "conditions": {
            "modality": "Image",
            "input": correct_cover['path'],
            "genre": correct_cover['genre']
        },
        "options": {
            "A": {"modality": "Text", "input": genre_options[0]},
            "B": {"modality": "Text", "input": genre_options[1]},
            "C": {"modality": "Text", "input": genre_options[2]},
            "D": {"modality": "Text", "input": genre_options[3]}
        },
        "correct_answer": correct_answer,
        "correct_genre": target_genre
    }
    return question


def generate_question_text_vision(covers, audios, correct_answer, target_genre):
    """Generate text -> vision question: read genre name and choose the matching album cover."""
    correct_idx = ord(correct_answer) - ord('A')
    
    question = {
        "question": f"Based on the music genre '{target_genre}', which album cover is most likely from this genre? Answer with A, B, C, or D",
        "target_genre": target_genre,
        "conditions": {
            "modality": "Text",
            "input": target_genre,
        },
        "options": {
            "A": {"modality": "Image", "input": covers[0]['path'], "genre": covers[0]['genre']},
            "B": {"modality": "Image", "input": covers[1]['path'], "genre": covers[1]['genre']},
            "C": {"modality": "Image", "input": covers[2]['path'], "genre": covers[2]['genre']},
            "D": {"modality": "Image", "input": covers[3]['path'], "genre": covers[3]['genre']}
        },
        "correct_answer": correct_answer,
        "correct_genre": target_genre
    }
    return question


def generate_question_text_audio(covers, audios, correct_answer, target_genre):
    """Generate text -> audio question: read genre name and choose the matching music."""
    correct_idx = ord(correct_answer) - ord('A')
    
    question = {
        "question": f"Based on the music genre '{target_genre}', which audio clip is from this genre? Answer with A, B, C, or D",
        "target_genre": target_genre,
        "conditions": {
            "modality": "Text",
            "input": target_genre,
        },
        "options": {
            "A": {"modality": "Audio", "input": audios[0]['path'], "genre": audios[0]['genre']},
            "B": {"modality": "Audio", "input": audios[1]['path'], "genre": audios[1]['genre']},
            "C": {"modality": "Audio", "input": audios[2]['path'], "genre": audios[2]['genre']},
            "D": {"modality": "Audio", "input": audios[3]['path'], "genre": audios[3]['genre']}
        },
        "correct_answer": correct_answer,
        "correct_genre": target_genre
    }
    return question


def generate_question_audio_text(covers, audios, correct_answer, target_genre):
    """Generate audio -> text question: listen to music and choose the matching genre name."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_audio = audios[correct_idx]
    
    # Get genre names from all options
    genre_options = [audios[i]['genre'] for i in range(4)]
    
    question = {
        "question": "Listen to this music clip. Which genre best describes the music you hear? Answer with A, B, C, or D",
        "target_genre": target_genre,
        "conditions": {
            "modality": "Audio",
            "input": correct_audio['path'],
            "genre": correct_audio['genre']
        },
        "options": {
            "A": {"modality": "Text", "input": genre_options[0]},
            "B": {"modality": "Text", "input": genre_options[1]},
            "C": {"modality": "Text", "input": genre_options[2]},
            "D": {"modality": "Text", "input": genre_options[3]}
        },
        "correct_answer": correct_answer,
        "correct_genre": target_genre
    }
    return question


def create_processed_audio_segments(audios, output_dir, target_duration=10):
    """Create high-quality 10-second audio segments from the sampled audios."""
    processed_audios = []
    
    for audio in audios:
        # Create output filename
        output_filename = f"{audio['name']}_10s.wav"
        output_path = os.path.join(output_dir, audio['genre'], output_filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Check if processed file already exists
        if os.path.exists(output_path):
            processed_audio = audio.copy()
            processed_audio['path'] = output_path
            processed_audios.append(processed_audio)
            continue
        
        # Extract high-quality segment
        audio_segment, start_time = extract_high_quality_audio_segment(
            audio['path'], target_duration
        )
        
        if audio_segment is not None:
            # Save the processed audio segment
            import soundfile as sf
            sf.write(output_path, audio_segment, 22050)
            
            # Create new audio instance with processed path
            processed_audio = audio.copy()
            processed_audio['path'] = output_path
            processed_audio['segment_start_time'] = start_time
            processed_audios.append(processed_audio)
        else:
            print(f"Failed to process audio for {audio['name']}")
            processed_audios.append(audio)  # Keep original if processing fails
    
    return processed_audios


def generate_all_modality_combinations(covers, audios, correct_answer, target_genre):
    """Generate all possible modality combinations for music genre classification."""
    questions = {}
    
    questions['audio_vision'] = generate_question_audio_vision(covers, audios, correct_answer, target_genre)
    questions['vision_audio'] = generate_question_vision_audio(covers, audios, correct_answer, target_genre)
    questions['vision_text'] = generate_question_vision_text(covers, audios, correct_answer, target_genre)
    questions['text_vision'] = generate_question_text_vision(covers, audios, correct_answer, target_genre)
    questions['text_audio'] = generate_question_text_audio(covers, audios, correct_answer, target_genre)
    questions['audio_text'] = generate_question_audio_text(covers, audios, correct_answer, target_genre)
    
    return questions


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    
    # Configuration parameters
    DATASET_NAME = 'gtzan'
    root_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/gtzan-dataset-music-genre-classification"  # Update to your actual data path
    processed_audio_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/gtzan-dataset-music-genre-classification/processed_audio_segments"  # Directory for 10-second segments
    
    N = 100  # Number of questions to generate per genre
    
    # Select modality combinations to generate
    GENERATE_COMBINATIONS = [
        'audio_vision',  # Audio -> Vision (Music -> Album Cover)
        'vision_audio',  # Vision -> Audio (Album Cover -> Music)
        'vision_text',   # Vision -> Text (Album Cover -> Genre)
        'text_vision',   # Text -> Vision (Genre -> Album Cover)
        'text_audio',    # Text -> Audio (Genre -> Music)
        'audio_text'     # Audio -> Text (Music -> Genre)
    ]
    
    export_dir = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/05_Exteral/music_genre_classification'
    
    # Process the music dataset
    print("Processing music dataset...")
    cover_groups, audio_groups = process_music_dataset(root_dir)
    
    print(f"Cover genres loaded:")
    for genre, instances in cover_groups.items():
        print(f"  {genre}: {len(instances)} covers")
    
    print(f"Audio genres loaded:")
    for genre, instances in audio_groups.items():
        print(f"  {genre}: {len(instances)} audio files")
    
    # Find common genres between covers and audio
    common_genres = set(cover_groups.keys()) & set(audio_groups.keys())
    valid_genres = [genre for genre in common_genres 
                   if len(cover_groups[genre]) >= 1 and len(audio_groups[genre]) >= 1]
    
    print(f"\nValid genres (have both covers and audio): {len(valid_genres)}")
    print(f"Valid genres: {valid_genres}")
    
    if len(valid_genres) < 4:
        print("Error: Need at least 4 genres to generate questions!")
        exit(1)
    
    # Initialize question lists for each combination and genre statistics
    all_questions = {combo: [] for combo in GENERATE_COMBINATIONS}
    genre_stats = {}
    
    # Create processed audio directory
    os.makedirs(processed_audio_dir, exist_ok=True)
    
    # Generate questions for each valid genre as the target (correct) genre
    for target_genre in valid_genres:
        print(f"\nGenerating questions with '{target_genre}' as target genre...")
        genre_questions = 0
        
        for i in range(N):
            # Sample mixed instances (1 correct + 3 distractors)
            sampled_covers, sampled_audios, correct_answer = sample_mixed_instances(
                cover_groups, audio_groups, target_genre, 4
            )
            
            if sampled_covers is None or sampled_audios is None:
                print(f"  Skipping iteration {i+1} for {target_genre}: sampling failed")
                continue
            
            # Create processed 10-second audio segments
            processed_audios = create_processed_audio_segments(
                sampled_audios, processed_audio_dir, target_duration=10
            )
            
            # Generate questions for all modality combinations
            questions = generate_all_modality_combinations(
                sampled_covers, processed_audios, correct_answer, target_genre
            )
            
            # Add to corresponding lists
            for combo in GENERATE_COMBINATIONS:
                if combo in questions:
                    all_questions[combo].append(questions[combo])
            
            genre_questions += 1
        
        genre_stats[target_genre] = genre_questions
        print(f"  Generated {genre_questions} questions with {target_genre} as target")
    
    # Save output files
    os.makedirs(export_dir, exist_ok=True)
    for combo in GENERATE_COMBINATIONS:
        filename = f"{export_dir}/{DATASET_NAME}_music_questions_{combo}.json"
        with open(filename, "w") as f:
            json.dump(all_questions[combo], f, indent=4)
        print(f"Saved {len(all_questions[combo])} questions to {filename}")
    
    # Save generation statistics
    stats = {
        "dataset": DATASET_NAME,
        "total_genres": len(valid_genres),
        "valid_genres": valid_genres,
        "questions_per_genre": N,
        "total_questions_per_combination": sum(len(questions) for questions in all_questions.values()) // len(GENERATE_COMBINATIONS),
        "genre_stats": genre_stats,
        "combinations": GENERATE_COMBINATIONS,
        "processed_audio_duration": 10,
        "audio_processing_method": "highest_energy_segment",
        "task_type": "genre_matching_across_modalities"
    }
    
    with open(f"{export_dir}/{DATASET_NAME}_generation_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    print(f"\n=== Generation Summary ===")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Valid genres: {len(valid_genres)}")
    print(f"Task: Genre matching across modalities")
    print(f"Questions per target genre: {N}")
    print(f"Audio segment duration: 10 seconds")
    print(f"Total questions per combination: {sum(len(questions) for questions in all_questions.values()) // len(GENERATE_COMBINATIONS)}")
    print(f"Generated combinations: {', '.join(GENERATE_COMBINATIONS)}")
    print(f"Export directory: {export_dir}")
    print(f"Processed audio directory: {processed_audio_dir}")
    
    print(f"\nTarget genre breakdown:")
    for genre, count in genre_stats.items():
        print(f"  {genre}: {count} questions")