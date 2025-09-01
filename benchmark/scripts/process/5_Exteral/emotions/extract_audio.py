import os
import pandas as pd
import cv2
import subprocess
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ==================== Video Processing and Audio Extraction for Emotion Analysis ====================


def parse_timestamp(timestamp_str):
    """
    Parse timestamp string to seconds.
    
    Args:
        timestamp_str (str): Timestamp in format "00:14:38,127" or "0:10:44,769"
        
    Returns:
        float: Time in seconds
    """
    # Clean up the timestamp string
    timestamp_str = timestamp_str.strip().strip('"')
    
    # Handle different formats
    if ',' in timestamp_str:
        time_part, ms_part = timestamp_str.split(',')
        ms = int(ms_part) / 1000.0
    else:
        time_part = timestamp_str
        ms = 0.0
    
    # Parse time part
    time_parts = time_part.split(':')
    if len(time_parts) == 3:
        hours, minutes, seconds = map(int, time_parts)
        total_seconds = hours * 3600 + minutes * 60 + seconds + ms
    elif len(time_parts) == 2:
        minutes, seconds = map(int, time_parts)
        total_seconds = minutes * 60 + seconds + ms
    else:
        total_seconds = float(time_parts[0]) + ms
    
    return total_seconds


def get_video_mapping(video_dir):
    """
    Create mapping between dialogue IDs and video files.
    
    Args:
        video_dir (str): Directory containing video files
        
    Returns:
        dict: Mapping of dialogue_id to video file path
    """
    video_mapping = {}
    
    if not os.path.exists(video_dir):
        print(f"Error: Video directory not found: {video_dir}")
        return video_mapping
    
    # Get all video files that start with 'dia'
    video_files = []
    for file in os.listdir(video_dir):
        if file.startswith('dia') and file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_files.append(file)
    
    # Sort by name to get correct order
    video_files.sort()
    
    # Map dialogue IDs to video files
    for i, video_file in enumerate(video_files):
        video_mapping[i] = os.path.join(video_dir, video_file)
    
    print(f"Found {len(video_files)} video files")
    return video_mapping


def add_subtitle_to_video(input_video, output_video, text, start_time, end_time):
    """
    Add subtitle to video using ffmpeg.
    
    Args:
        input_video (str): Input video path
        output_video (str): Output video path
        text (str): Subtitle text
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
    """
    # Create subtitle file
    subtitle_file = output_video.replace('.mp4', '.srt')
    
    # Clean text for subtitle
    clean_text = text.replace('"', '').replace('\n', ' ').strip()
    
    # Create SRT content
    srt_content = f"""1
{format_time_srt(0)} --> {format_time_srt(end_time-start_time)}
{clean_text}

"""
    
    with open(subtitle_file, 'w', encoding='utf-8') as f:
        f.write(srt_content)
    
    # FFmpeg command to add subtitle
    cmd = [
        'ffmpeg', '-i', input_video,
        '-vf', f"subtitles={subtitle_file}:force_style='Alignment=2,Fontsize=20,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2'",
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'medium',
        '-an',  # Remove audio
        '-y',   # Overwrite output
        output_video
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    # Clean up subtitle file
    os.remove(subtitle_file)
    return True



def format_time_srt(seconds):
    """Format time for SRT subtitle format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def extract_audio_from_video(input_video, output_audio, start_time, end_time):
    """
    Extract audio segment from video.
    
    Args:
        input_video (str): Input video path
        output_audio (str): Output audio path
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
    """
    duration = end_time - start_time
    
    cmd = [
        'ffmpeg',
        '-i', input_video,
        '-ss', str(0),
        '-t', str(duration),
        '-vn',  # No video
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',  # Mono
        '-y',   # Overwrite output
        output_audio
    ]
    # import ipdb; ipdb.set_trace()
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        return False


def process_emotion_dataset(csv_path, video_dir, output_dir):
    """
    Process the emotion dataset to create video clips with subtitles and audio files.
    
    Args:
        csv_path (str): Path to the CSV file
        video_dir (str): Directory containing video files
        output_dir (str): Output directory for processed files
    """
    # Load CSV data
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records from CSV")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # Get video mapping
    video_mapping = get_video_mapping(video_dir)
    
    if not video_mapping:
        print("No video files found!")
        return
    
    # Process each record
    processed_count = 0
    error_count = 0
    
    for idx, row in tqdm(df.iterrows()):

        # Extract information
        dialogue_id = int(row['Dialogue_ID'])
        utterance_id = int(row['Utterance_ID'])
        sentiment = row['Sentiment']
        emotion = row['Emotion']
        utterance = row['Utterance']
        start_time = parse_timestamp(row['StartTime'])
        end_time = parse_timestamp(row['EndTime'])
        
        
        input_video = os.path.join(video_dir, f"dia{dialogue_id}_utt{utterance_id}.mp4")
        
        # Create output directory structure: sentiment/emotion/
        output_sentiment_dir = os.path.join(output_dir, sentiment)
        output_emotion_dir = os.path.join(output_sentiment_dir, emotion)
        os.makedirs(output_emotion_dir, exist_ok=True)
        
        # Create output filenames
        video_filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        audio_filename = f"dia{dialogue_id}_utt{utterance_id}.wav"
        
        output_video = os.path.join(output_emotion_dir, video_filename)
        output_audio = os.path.join(output_emotion_dir, audio_filename)
        
        # Skip if files already exist
        # if os.path.exists(output_video) and os.path.exists(output_audio):
        #     processed_count += 1
        #     continue
        
        # Add subtitle to video (without audio)
        print(f"Processing {dialogue_id}_{utterance_id}: {emotion} ({sentiment})")
        if add_subtitle_to_video(input_video, output_video, utterance, start_time, end_time):
            print(f"  Successfully processed: {video_filename}")

        else:
            error_count += 1
            print(f"  Failed to process video: {video_filename}")
            
        if os.path.exists(output_video) and extract_audio_from_video(input_video, output_audio, start_time, end_time):
            processed_count += 1
            print(f"  Successfully processed: {video_filename}")
        else:
            error_count += 1
            print(f"  Failed to extract audio for: {video_filename}")
        
        # Progress update
        if (processed_count + error_count) % 50 == 0:
            print(f"Progress: {processed_count} processed, {error_count} errors")
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Total records: {len(df)}")


if __name__ == "__main__":
    # Configuration
    csv_path = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/emotions/MELD.Raw/test_sent_emo.csv"  # Update this path
    video_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/emotions/MELD.Raw/output_repeated_splits_test"
    output_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/emotions/processed_emotion_data"  # Update this path
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("FFmpeg is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: FFmpeg is not installed or not in PATH")
        exit(1)
    
    # Process the dataset
    process_emotion_dataset(csv_path, video_dir, output_dir)
    
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    ├── positive/")
    print(f"    │   ├── joy/")
    print(f"    │   │   ├── dia0_utt1.mp4 (video with subtitles)")
    print(f"    │   │   └── dia0_utt1.wav (audio)")
    print(f"    │   └── surprise/")
    print(f"    ├── negative/")
    print(f"    │   ├── anger/")
    print(f"    │   └── sadness/")
    print(f"    └── neutral/")
    print(f"        └── neutral/")