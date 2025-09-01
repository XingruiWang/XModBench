import os
import re
import random
import json
from pathlib import Path
import glob
import librosa
import numpy as np
from scipy.signal import find_peaks
import soundfile as sf


# ==================== Singer Identification Multimodal Question-Answer Generator ====================

def extract_singer_info_from_path(audio_path):
    """
    Extract singer and track information from the audio file path.
    
    Args:
        audio_path (str): Path to the audio file
        
    Returns:
        dict: Dictionary containing extracted information
    """
    try:
        # Get the filename without extension
        filename = os.path.basename(audio_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Get the parent directory name (should be the singer name)
        parent_dir = os.path.basename(os.path.dirname(os.path.dirname(audio_path)))
        
        # Try to extract track ID from filename
        # Common patterns: "track_001.wav", "song_1.mp3", "artist_song_title.wav", etc.
        track_id = None
        
        # Pattern 1: Look for numbers in the filename
        import re
        number_match = re.search(r'(\d+)', name_without_ext)
        if number_match:
            track_id = f"track_{number_match.group(1).zfill(3)}"
        else:
            # Pattern 2: Use filename as track ID (cleaned)
            # Remove common prefixes/suffixes and clean the name
            cleaned_name = re.sub(r'[^\w\s-]', '', name_without_ext)
            cleaned_name = re.sub(r'\s+', '_', cleaned_name.strip())
            track_id = cleaned_name[:50] if cleaned_name else "unknown_track"
        
        # Extract additional metadata if available from filename
        # Look for common separators like "_", "-", or spaces
        parts = re.split(r'[-_\s]+', name_without_ext)
        
        info = {
            'track_id': track_id,
            'filename': filename,
            'singer_dir': parent_dir,
            'filename_parts': parts,
            'original_filename': name_without_ext
        }
        
        return info
        
    except Exception as e:
        # Fallback in case of any errors
        filename = os.path.basename(audio_path) if audio_path else "unknown"
        return {
            'track_id': f"track_{hash(filename) % 10000:04d}",
            'filename': filename,
            'singer_dir': 'unknown',
            'filename_parts': [filename],
            'original_filename': os.path.splitext(filename)[0] if filename else "unknown"
        }
        

def extract_representative_audio_segment(audio_path, target_duration=10, sr=22050):
    """
    Extract a representative 10-second segment from audio (focusing on vocal highlights/chorus).
    
    Args:
        audio_path (str): Path to the audio file
        target_duration (int): Target duration in seconds (default: 10)
        sr (int): Sample rate (default: 22050)
        
    Returns:
        tuple: (audio_segment, start_time, analysis_info) or (None, None, None) if extraction fails
    """
    try:
        # Load audio file
        y, actual_sr = librosa.load(audio_path, sr=sr)
        total_duration = len(y) / actual_sr
        
        if total_duration < target_duration:
            print(f"Warning: Audio {audio_path} is shorter than {target_duration} seconds")
            return y, 0, {"reason": "full_audio", "duration": total_duration}
        
        print(f"Analyzing audio: {os.path.basename(audio_path)} ({total_duration:.1f}s)")
        
        # Multiple analysis methods to find the best segment
        
        # Method 1: RMS Energy Analysis (find energetic parts)
        hop_length = 512
        frame_length = 2048
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Method 2: Spectral Centroid (find parts with rich harmonics)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=actual_sr, hop_length=hop_length)[0]
        
        # Method 3: Tempo and Beat tracking (find rhythmically strong parts)
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=actual_sr, hop_length=hop_length)
            beat_strength = np.zeros_like(rms)
            if len(beats) > 0:
                beat_frames = librosa.util.fix_length(beats, size=len(rms))
                beat_strength = np.histogram(beat_frames, bins=len(rms), range=(0, len(rms)))[0]
        except:
            beat_strength = np.zeros_like(rms)
        
        # Method 4: Harmonic-percussive separation (find vocal-prominent parts)
        try:
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_rms = librosa.feature.rms(y=y_harmonic, hop_length=hop_length)[0]
        except:
            harmonic_rms = rms
        
        # Method 5: MFCC variance (find diverse/interesting audio parts)
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=actual_sr, n_mfcc=13, hop_length=hop_length)
            mfcc_variance = np.var(mfccs, axis=0)
        except:
            mfcc_variance = np.ones_like(rms)
        
        # Normalize all features
        def normalize_feature(feature):
            if np.std(feature) == 0:
                return np.ones_like(feature)
            return (feature - np.mean(feature)) / np.std(feature)
        
        rms_norm = normalize_feature(rms)
        centroid_norm = normalize_feature(spectral_centroid)
        beat_norm = normalize_feature(beat_strength)
        harmonic_norm = normalize_feature(harmonic_rms)
        mfcc_norm = normalize_feature(mfcc_variance)
        
        # Combined score emphasizing vocal and energetic parts
        combined_score = (
            0.3 * rms_norm +           # Energy
            0.2 * centroid_norm +      # Spectral richness  
            0.2 * beat_norm +          # Rhythmic strength
            0.2 * harmonic_norm +      # Harmonic content (vocals)
            0.1 * mfcc_norm            # Timbral diversity
        )
        
        # Apply smoothing to avoid very short peaks
        from scipy import ndimage
        combined_score = ndimage.gaussian_filter1d(combined_score, sigma=2)
        
        # Find the best segment
        segment_length_frames = int(target_duration * actual_sr / hop_length)
        
        if segment_length_frames >= len(combined_score):
            return y, 0, {"reason": "full_audio", "duration": total_duration}
        
        # Find peaks in the combined score
        peaks, properties = find_peaks(combined_score, height=np.mean(combined_score), 
                                     distance=segment_length_frames//2)
        
        if len(peaks) > 0:
            # Choose the highest peak that allows for a full segment
            valid_peaks = peaks[peaks <= len(combined_score) - segment_length_frames]
            if len(valid_peaks) > 0:
                best_peak = valid_peaks[np.argmax(combined_score[valid_peaks])]
                best_start_frame = max(0, best_peak - segment_length_frames//2)
            else:
                # Fallback: highest score in valid range
                valid_scores = combined_score[:len(combined_score) - segment_length_frames]
                best_start_frame = np.argmax(valid_scores)
        else:
            # Fallback: highest average score over segment length
            best_start_frame = 0
            best_score = 0
            for i in range(len(combined_score) - segment_length_frames):
                segment_score = np.mean(combined_score[i:i + segment_length_frames])
                if segment_score > best_score:
                    best_score = segment_score
                    best_start_frame = i
        
        # Convert back to audio samples
        start_sample = int(best_start_frame * hop_length)
        end_sample = start_sample + int(target_duration * actual_sr)
        end_sample = min(end_sample, len(y))
        
        audio_segment = y[start_sample:end_sample]
        start_time = start_sample / actual_sr
        
        # Analysis info for debugging
        analysis_info = {
            "reason": "intelligent_extraction",
            "start_time": start_time,
            "duration": len(audio_segment) / actual_sr,
            "peak_energy": float(np.max(rms)),
            "avg_spectral_centroid": float(np.mean(spectral_centroid)),
            "estimated_tempo": float(tempo) if 'tempo' in locals() else 0,
            "segment_score": float(np.mean(combined_score[best_start_frame:best_start_frame + segment_length_frames]))
        }
        
        print(f"  Selected segment: {start_time:.1f}s - {start_time + target_duration:.1f}s")
        
        return audio_segment, start_time, analysis_info
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None, None


def process_singer_dataset(singers_dir):
    """
    Process the singer dataset and organize by singers.
    
    Args:
        singers_dir (str): Root directory containing singers data
        
    Returns:
        dict: Dictionary with singer names as keys and their data as values
    """
    singer_groups = {}
    
    if not os.path.exists(singers_dir):
        print(f"Error: Singers directory not found: {singers_dir}")
        return {}
    
    # Process each singer folder
    for singer_folder in os.listdir(singers_dir):
        singer_path = os.path.join(singers_dir, singer_folder)
        if not os.path.isdir(singer_path):
            continue
        
        singer_name = singer_folder
        singer_groups[singer_name] = {
            'name': singer_name,
            'audio_files': [],
            'images': []
        }
        
        # Process audio files (only full tracks, not segments)
        audio_dir = os.path.join(singer_path, 'audio')
        if os.path.exists(audio_dir):
            for audio_file in os.listdir(audio_dir):
                if audio_file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', 'webm')):
                    # Skip pre-segmented files
                    if '_seg' in audio_file.lower():
                        continue
                        
                    audio_path = os.path.join(audio_dir, audio_file)
                    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                        # Validate audio file
                        try:
                            import soundfile as sf
                            info = sf.info(audio_path)
                            if info.frames > 0 and info.frames / info.samplerate >= 10:  # At least 10 seconds
                                audio_info = extract_singer_info_from_path(audio_path)
                                singer_groups[singer_name]['audio_files'].append({
                                    'path': audio_path,
                                    'filename': audio_file,
                                    'track_id': audio_info['track_id'],
                                    'duration': info.frames / info.samplerate,
                                    'is_full_track': True
                                })
                            else:
                                print(f"Warning: Audio file too short: {audio_path}")
                        except Exception as e:
                            print(f"Warning: Cannot validate audio file {audio_path}: {e}")
        
        # Process image files
        images_dir = os.path.join(singer_path, 'images')
        if os.path.exists(images_dir):
            for img_file in os.listdir(images_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(images_dir, img_file)
                    if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                        singer_groups[singer_name]['images'].append({
                            'path': img_path,
                            'filename': img_file
                        })
    
    # Filter out singers without sufficient data
    valid_singers = {}
    for singer_name, data in singer_groups.items():
        if len(data['audio_files']) > 0 and len(data['images']) > 0:
            valid_singers[singer_name] = data
            print(f"Singer '{singer_name}': {len(data['audio_files'])} full audio tracks, {len(data['images'])} images")
    
    return valid_singers


def create_processed_audio_segments(singers, output_dir, target_duration=10):
    """Create high-quality 10-second audio segments from the sampled singers."""
    processed_singers = []
    
    for singer in singers:
        # Create output filename
        original_filename = singer['audio']['filename']
        output_filename = f"{singer['singer_name']}_{singer['audio']['track_id']}_best10s.wav"
        singer_output_dir = os.path.join(output_dir, singer['singer_name'])
        output_path = os.path.join(singer_output_dir, output_filename)
        
        # Create directory if it doesn't exist
        os.makedirs(singer_output_dir, exist_ok=True)
        
        # Check if processed file already exists
        if os.path.exists(output_path):
            processed_singer = singer.copy()
            processed_singer['audio'] = singer['audio'].copy()
            processed_singer['audio']['path'] = output_path
            processed_singer['audio']['is_processed_segment'] = True
            processed_singers.append(processed_singer)
            continue
        
        
        audio_segment, start_time, analysis_info = extract_representative_audio_segment(
            singer['audio']['path'], target_duration
        )
        
        if audio_segment is not None:
            # Save the processed audio segment
            sf.write(output_path, audio_segment, 22050)
            
            # Create new singer instance with processed audio path
            processed_singer = singer.copy()
            processed_singer['audio'] = singer['audio'].copy()
            processed_singer['audio']['path'] = output_path
            processed_singer['audio']['segment_start_time'] = start_time
            processed_singer['audio']['analysis_info'] = analysis_info
            processed_singer['audio']['is_processed_segment'] = True
            processed_singers.append(processed_singer)
        else:
            print(f"Failed to process audio for {singer['singer_name']}")
            # Keep original if processing fails
            processed_singers.append(singer)
    
    return processed_singers


def sample_mixed_singer_instances(singer_groups, target_singer, n_samples=4):
    """
    Sample singer instances with one correct target and distractors.
    
    Args:
        singer_groups (dict): Dictionary of all singer data
        target_singer (str): The correct singer name for this question
        n_samples (int): Total number of options (default: 4)
        
    Returns:
        tuple: (sampled_data, correct_answer) or (None, None) if sampling fails
    """
    if target_singer not in singer_groups or len(singer_groups) < n_samples:
        return None, None
    
    # Get target singer data
    target_data = singer_groups[target_singer]
    
    # Sample distractor singers
    other_singers = [name for name in singer_groups.keys() if name != target_singer]
    if len(other_singers) < n_samples - 1:
        return None, None
    
    distractor_singers = random.sample(other_singers, n_samples - 1)
    selected_singers = [target_singer] + distractor_singers
    random.shuffle(selected_singers)
    
    # Find the correct answer position
    correct_idx = selected_singers.index(target_singer)
    correct_answer = chr(ord('A') + correct_idx)
    
    # Sample data for each selected singer
    sampled_data = []
    for singer_name in selected_singers:
        singer_data = singer_groups[singer_name]
        
        # Sample audio file (only full tracks)
        full_track_files = [af for af in singer_data['audio_files'] if af['is_full_track']]
        if not full_track_files:
            return None, None
        
        selected_audio = random.choice(full_track_files)
        
        # Sample image file
        if not singer_data['images']:
            return None, None
        
        selected_image = random.choice(singer_data['images'])
        
        sampled_data.append({
            'singer_name': singer_name,
            'audio': selected_audio,
            'image': selected_image
        })
    
    return sampled_data, correct_answer


def generate_question_audio_vision(singers, correct_answer, target_singer_name):
    """Generate audio -> vision question: listen to song and choose singer's photo."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_singer = singers[correct_idx]
    
    question = {
        "question": "Listen to this song. Which photo shows the singer performing this music? Answer with A, B, C, or D",
        "target_singer": target_singer_name,
        "conditions": {
            "modality": "Audio",
            "input": correct_singer['audio']['path'],
            "singer": correct_singer['singer_name'],
            "track_id": correct_singer['audio']['track_id'],
           
        },
        "options": {
            "A": {"modality": "Image", "input": singers[0]['image']['path'], "singer": singers[0]['singer_name']},
            "B": {"modality": "Image", "input": singers[1]['image']['path'], "singer": singers[1]['singer_name']},
            "C": {"modality": "Image", "input": singers[2]['image']['path'], "singer": singers[2]['singer_name']},
            "D": {"modality": "Image", "input": singers[3]['image']['path'], "singer": singers[3]['singer_name']}
        },
        "correct_answer": correct_answer,
        "correct_singer": target_singer_name
    }
    return question


def generate_question_vision_audio(singers, correct_answer, target_singer_name):
    """Generate vision -> audio question: look at singer photo and choose their song."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_singer = singers[correct_idx]
    
    question = {
        "question": "Look at this singer's photo. Which song is performed by this artist? Answer with A, B, C, or D",
        "target_singer": target_singer_name,
        "conditions": {
            "modality": "Image",
            "input": correct_singer['image']['path'],
            "singer": correct_singer['singer_name']
        },
        "options": {
            "A": {"modality": "Audio", "input": singers[0]['audio']['path'], "singer": singers[0]['singer_name'], "track_id": singers[0]['audio']['track_id']},
            "B": {"modality": "Audio", "input": singers[1]['audio']['path'], "singer": singers[1]['singer_name'], "track_id": singers[1]['audio']['track_id']},
            "C": {"modality": "Audio", "input": singers[2]['audio']['path'], "singer": singers[2]['singer_name'], "track_id": singers[2]['audio']['track_id']},
            "D": {"modality": "Audio", "input": singers[3]['audio']['path'], "singer": singers[3]['singer_name'], "track_id": singers[3]['audio']['track_id']}
        },
        "correct_answer": correct_answer,
        "correct_singer": target_singer_name
    }
    return question


def generate_question_vision_text(singers, correct_answer, target_singer_name):
    """Generate vision -> text question: look at singer photo and choose singer name."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_singer = singers[correct_idx]
    
    # Get singer names from all options
    singer_names = [singer['singer_name'] for singer in singers]
    
    question = {
        "question": "Look at this photo. Which singer is shown in this image? Answer with A, B, C, or D",
        "target_singer": target_singer_name,
        "conditions": {
            "modality": "Image",
            "input": correct_singer['image']['path'],
            "singer": correct_singer['singer_name']
        },
        "options": {
            "A": {"modality": "Text", "input": singer_names[0].replace('_', ' ')},
            "B": {"modality": "Text", "input": singer_names[1].replace('_', ' ')},
            "C": {"modality": "Text", "input": singer_names[2].replace('_', ' ')},
            "D": {"modality": "Text", "input": singer_names[3].replace('_', ' ')}
        },
        "correct_answer": correct_answer,
        "correct_singer": target_singer_name
    }
    return question


def generate_question_text_vision(singers, correct_answer, target_singer_name):
    """Generate text -> vision question: read singer name and choose their photo."""
    correct_idx = ord(correct_answer) - ord('A')
    
    question = {
        "question": f"Based on the singer name '{target_singer_name.replace('_', ' ')}', which photo shows this artist? Answer with A, B, C, or D",
        "target_singer": target_singer_name,
        "conditions": {
            "modality": "Text",
            "input": target_singer_name.replace('_', ' ')
        },
        "options": {
            "A": {"modality": "Image", "input": singers[0]['image']['path'], "singer": singers[0]['singer_name']},
            "B": {"modality": "Image", "input": singers[1]['image']['path'], "singer": singers[1]['singer_name']},
            "C": {"modality": "Image", "input": singers[2]['image']['path'], "singer": singers[2]['singer_name']},
            "D": {"modality": "Image", "input": singers[3]['image']['path'], "singer": singers[3]['singer_name']}
        },
        "correct_answer": correct_answer,
        "correct_singer": target_singer_name
    }
    return question


def generate_question_text_audio(singers, correct_answer, target_singer_name):
    """Generate text -> audio question: read singer name and choose their song."""
    correct_idx = ord(correct_answer) - ord('A')
    
    question = {
        "question": f"Based on the singer name '{target_singer_name.replace('_', ' ')}', which song is performed by this artist? Answer with A, B, C, or D",
        "target_singer": target_singer_name,
        "conditions": {
            "modality": "Text",
            "input": target_singer_name.replace('_', ' ')
        },
        "options": {
            "A": {"modality": "Audio", "input": singers[0]['audio']['path'], "singer": singers[0]['singer_name'], "track_id": singers[0]['audio']['track_id']},
            "B": {"modality": "Audio", "input": singers[1]['audio']['path'], "singer": singers[1]['singer_name'], "track_id": singers[1]['audio']['track_id']},
            "C": {"modality": "Audio", "input": singers[2]['audio']['path'], "singer": singers[2]['singer_name'], "track_id": singers[2]['audio']['track_id']},
            "D": {"modality": "Audio", "input": singers[3]['audio']['path'], "singer": singers[3]['singer_name'], "track_id": singers[3]['audio']['track_id']}
        },
        "correct_answer": correct_answer,
        "correct_singer": target_singer_name
    }
    return question


def generate_question_audio_text(singers, correct_answer, target_singer_name):
    """Generate audio -> text question: listen to song and choose singer name."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_singer = singers[correct_idx]
    
    # Get singer names from all options
    singer_names = [singer['singer_name'] for singer in singers]
    
    question = {
        "question": "Listen to this song. Which artist is performing this music? Answer with A, B, C, or D",
        "target_singer": target_singer_name,
        "conditions": {
            "modality": "Audio",
            "input": correct_singer['audio']['path'],
            "singer": correct_singer['singer_name'],
            "track_id": correct_singer['audio']['track_id'],
           
        },
        "options": {
            "A": {"modality": "Text", "input": singer_names[0].replace('_', ' ')},
            "B": {"modality": "Text", "input": singer_names[1].replace('_', ' ')},
            "C": {"modality": "Text", "input": singer_names[2].replace('_', ' ')},
            "D": {"modality": "Text", "input": singer_names[3].replace('_', ' ')}
        },
        "correct_answer": correct_answer,
        "correct_singer": target_singer_name
    }
    return question


def generate_all_modality_combinations(singers, correct_answer, target_singer_name):
    """Generate all possible modality combinations for singer identification."""
    questions = {}
    
    questions['audio_vision'] = generate_question_audio_vision(singers, correct_answer, target_singer_name)
    questions['vision_audio'] = generate_question_vision_audio(singers, correct_answer, target_singer_name)
    questions['vision_text'] = generate_question_vision_text(singers, correct_answer, target_singer_name)
    questions['text_vision'] = generate_question_text_vision(singers, correct_answer, target_singer_name)
    questions['text_audio'] = generate_question_text_audio(singers, correct_answer, target_singer_name)
    questions['audio_text'] = generate_question_audio_text(singers, correct_answer, target_singer_name)
    
    return questions


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    
    # Configuration parameters
    DATASET_NAME = 'singer_identification'
    singers_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/process/5_Exteral/singers_data"  # Update to your singers directory path
    processed_audio_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/process/5_Exteral/singers_data_processed"  # Directory for 10-second best segments
    
    N = 150  # Number of questions to generate
    
    # Select modality combinations to generate
    GENERATE_COMBINATIONS = [
        'audio_vision',  # Song -> Photo (Listen to song, choose singer photo)
        'vision_audio',  # Photo -> Song (Look at singer, choose their song)
        'vision_text',   # Photo -> Name (Look at singer, choose their name)
        'text_vision',   # Name -> Photo (Read name, choose singer photo)
        'text_audio',    # Name -> Song (Read name, choose their song)
        'audio_text'     # Song -> Name (Listen to song, choose singer name)
    ]
    
    export_dir = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/singer_identification/{DATASET_NAME}'
    
    # Process the singer dataset
    print("Processing singer dataset...")
    singer_groups = process_singer_dataset(singers_dir)
    
    if len(singer_groups) < 4:
        print("Error: Need at least 4 singers with complete data to generate questions!")
        exit(1)
    
    print(f"\nDataset loaded successfully:")
    print(f"  Total singers: {len(singer_groups)}")
    
    # Sample some singer names for preview
    sample_singers = list(singer_groups.keys())[:10]
    print(f"  Sample singers: {sample_singers}")
    
    # Initialize question lists for each combination
    all_questions = {combo: [] for combo in GENERATE_COMBINATIONS}
    singer_stats = {}
    
    # Create processed audio directory
    os.makedirs(processed_audio_dir, exist_ok=True)
    
    # Generate questions using different target singers
    singer_names = list(singer_groups.keys())
    successful_generations = 0
    
    for i in range(N):
        # Randomly select a target singer
        target_singer_name = random.choice(singer_names)
        
        # Sample mixed instances (1 correct + 3 distractors)
        sampled_singers, correct_answer = sample_mixed_singer_instances(
            singer_groups, target_singer_name, 4
        )
        
        if sampled_singers is None:
            print(f"  Skipping iteration {i+1}: sampling failed for {target_singer_name}")
            continue
        
        # Create processed 10-second representative audio segments
        processed_singers = create_processed_audio_segments(
            sampled_singers, processed_audio_dir, target_duration=30
        )
        
        if len(processed_singers) < 4:
            print(f"  Skipping iteration {i+1}: audio processing failed")
            continue
        
        # Generate questions for all modality combinations
        questions = generate_all_modality_combinations(
            processed_singers, correct_answer, target_singer_name
        )
        
        # Add to corresponding lists
        for combo in GENERATE_COMBINATIONS:
            if combo in questions:
                all_questions[combo].append(questions[combo])
        
        successful_generations += 1
        
        # Track which singers were used as targets
        if target_singer_name not in singer_stats:
            singer_stats[target_singer_name] = 0
        singer_stats[target_singer_name] += 1
        
        if (i + 1) % 30 == 0:
            print(f"  Generated {successful_generations} successful questions out of {i + 1} attempts")
    
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
        "total_singers": len(singer_groups),
        "questions_generated": successful_generations,
        "total_questions_per_combination": successful_generations,
        "singer_usage_stats": singer_stats,
        "combinations": GENERATE_COMBINATIONS,
        "task_type": "singer_identification_across_modalities",
        "processed_audio_duration": 10,
        "audio_processing_method": "intelligent_vocal_highlight_extraction",
        "data_sources": {
            "singers_dir": singers_dir,
            "processed_audio_dir": processed_audio_dir
        }
    }
    
    with open(f"{export_dir}/{DATASET_NAME}_generation_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    print(f"\n=== Generation Summary ===")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Total singers: {len(singer_groups)}")
    print(f"Task: Singer identification across modalities")
    print(f"Successful questions generated: {successful_generations}")
    print(f"Questions per combination: {successful_generations}")
    print(f"Generated combinations: {', '.join(GENERATE_COMBINATIONS)}")
    print(f"Export directory: {export_dir}")
    
    print(f"\nTop 10 most frequently used target singers:")
    sorted_singers = sorted(singer_stats.items(), key=lambda x: x[1], reverse=True)[:10]
    for singer_name, count in sorted_singers:
        print(f"  {singer_name.replace('_', ' ')}: {count} times")