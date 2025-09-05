import cv2
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import math
import random
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import json
import os
import subprocess
from rotate_audio import rotate_audio_azimuth_mic
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
from tqdm import tqdm
import time


class STARSS23QuestionGenerator:
    def __init__(self):
        # Sound event class mapping
        self.sound_classes = {
            0: "Female speech",
            1: "Male speech", 
            2: "Clapping",
            3: "Telephone",
            4: "Laughter",
            5: "Domestic sounds",
            6: "Walk, footsteps",
            7: "Door, open or close",
            8: "Music",
            9: "Musical instrument",
            10: "Water tap, faucet",
            11: "Bell",
            12: "Knock"
        }
        
        # Frame rate for metadata (100ms = 0.1s per frame)
        self.metadata_fps = 10  # 10 frames per second
        
    def load_metadata(self, csv_path: str) -> pd.DataFrame:
        """Load and parse the CSV metadata file"""
        data = []
        
        with open(csv_path, 'r') as f:
            content = f.read().strip()
            
        # Split by spaces and parse each entry
        entries = content.split()
        for entry in entries:
            if ',' in entry:
                parts = entry.split(',')
                if len(parts) == 6:
                    frame, class_idx, source_idx, azimuth, elevation, distance = map(int, parts)
                    data.append({
                        'frame': frame,
                        'class_idx': class_idx,
                        'source_idx': source_idx,
                        'azimuth': azimuth,
                        'elevation': elevation,
                        'distance': distance,
                        'class_name': self.sound_classes[class_idx]
                    })
        
        return pd.DataFrame(data)
    
    def check_event_processed(self, output_dir: str, question_id: str, question_type: str) -> bool:
        """Check if an event has already been processed"""
        output_path = Path(output_dir)
        metadata_file = output_path / f"{question_id}_metadata.json"
        
        # Check if metadata file exists
        if not metadata_file.exists():
            return False
        
        # Check if all required files exist
        if question_type == 'video_choice':
            required_files = [
                f"{question_id}_input_audio.wav",
                f"{question_id}_choice_0.mp4",
                f"{question_id}_choice_1.mp4", 
                f"{question_id}_choice_2.mp4",
                f"{question_id}_choice_3.mp4",
                f"{question_id}_visualization.png",
                f"{question_id}_metadata.json"
            ]
        else:  # audio_choice
            required_files = [
                f"{question_id}_input_video.mp4",
                f"{question_id}_choice_0.wav",
                f"{question_id}_choice_1.wav",
                f"{question_id}_choice_2.wav", 
                f"{question_id}_choice_3.wav",
                f"{question_id}_visualization.png",
                f"{question_id}_metadata.json"
            ]
        
        # Check if all files exist and are not empty
        for file_name in required_files:
            file_path = output_path / file_name
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False
        
        return True
    
    def load_audio(self, audio_path: str, sr: int = 24000) -> Tuple[np.ndarray, int]:
        """Load multichannel audio file"""
        audio, sample_rate = librosa.load(audio_path, sr=sr, mono=False)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        return audio, sample_rate
    
    def extract_audio_segment(self, audio: np.ndarray, start_time: float, 
                            duration: float, sample_rate: int) -> np.ndarray:
        """Extract audio segment for a specific time window"""
        start_sample = int(start_time * sample_rate)
        end_sample = int((start_time + duration) * sample_rate)
        return audio[:, start_sample:end_sample]
    
    def rotate_audio_azimuth_mic(self,audio_foa: np.ndarray, rotation_degrees: float) -> np.ndarray:
        mic_dirs = np.array([
                                [ 0.78539816,  0.61086524],
                                [-0.78539816, -0.61086524],
                                [ 2.35619449, -0.61086524],
                                [-2.35619449,  0.61086524]
                            ])
        
        return rotate_audio_azimuth_mic(audio_foa, mic_dirs, rotation_degrees)
    
    def rotate_audio_azimuth(self,audio_foa: np.ndarray, rotation_degrees: float) -> np.ndarray:
        """
        Rotate FOA with channel order [H1, H2, H3, H4] = [W, Y, Z, X]
        around the vertical (azimuth) axis.

        Convention: positive rotation_degrees = counter-clockwise (CCW).

        Parameters
        ----------
        audio_foa : np.ndarray
            Shape [4, T], channels-first, order [W, Y, Z, X].
        rotation_degrees : float
            CCW rotation angle in degrees.

        Returns
        -------
        rotated : np.ndarray
            Same shape as input.
        """
        if audio_foa.ndim != 2 or audio_foa.shape[0] < 4:
            raise ValueError("Expect FOA array [4, T] with order [W, Y, Z, X].")

        rotated = audio_foa.copy()

        theta = np.radians(rotation_degrees)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Indices under your format: W=0, Y=1, Z=2, X=3
        W, Y, Z, X = 0, 1, 2, 3
        rms_in = np.sqrt(np.mean(audio_foa**2) + 1e-12)

        X_in = audio_foa[X, :].copy()
        Y_in = audio_foa[Y, :].copy()

        # CCW rotation in XY-plane:
        # [X']   [ cos  -sin ] [X]
        # [Y'] = [ sin   cos ] [Y]
        rotated[X, :] =  cos_t * X_in - sin_t * Y_in
        rotated[Y, :] =  sin_t * X_in + cos_t * Y_in

        # W, Z unchanged for pure azimuth rotation
        rotated[W, :] = audio_foa[W, :]
        rotated[Z, :] = audio_foa[Z, :]


        # --- 2) 通道增益（可选微调方向感） ---
        gW=0.6
        gXY=1.20
        gZ=1.20
        headroom_db=1.0
        
        rotated[W, :] *= gW
        rotated[X, :] *= gXY
        rotated[Y, :] *= gXY
        rotated[Z, :] *= gZ

        # --- 3) 响度归一化 + 预留余量 ---
        rms_out = np.sqrt(np.mean(rotated**2) + 1e-12)
        if rms_out > 0:
            target = rms_in * (10 ** (-headroom_db / 20.0))  # 留 headroom
            rotated *= (target / rms_out)

        # （可选）硬限幅，确保绝对不削波
        peak = np.max(np.abs(rotated))
        if peak > 0.999:
            rotated *= (0.999 / peak)
        return rotated
    
    def create_centered_360_view(self, frame: np.ndarray, center_azimuth: float, 
                               output_width: int = 800, output_height: int = 400) -> np.ndarray:
        """
        Create a 360° view by horizontally shifting the equirectangular image 
        so that center_azimuth appears in the middle
        """
        height, width = frame.shape[:2]
        # Convert azimuth to pixel offset
        # Azimuth 0° should be at width/2, azimuth 180° at 0, azimuth -180° at width
        azimuth_normalized = (center_azimuth % 360) / 360.0  # 0 to 1
        pixel_offset = int(azimuth_normalized * width)
        
        # Create shifted image by rolling horizontally
        shifted_frame = np.roll(frame, -pixel_offset, axis=1)
        
        # Resize to desired output size
        resized_frame = cv2.resize(shifted_frame, (output_width, output_height))
        
        return resized_frame
    
    def extract_video_segment(self, video_path: str, start_time: float, duration: float,
                            center_azimuth: float, output_size: Tuple[int, int] = (1920, 960)) -> List[np.ndarray]:
        """Extract video segment with specific center azimuth"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame range
        start_frame = max(0, int((start_time - duration/2) * fps))
        end_frame = int((start_time + duration/2) * fps)
        
        frames = []
        for frame_idx in range(start_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Apply 360° transformation with center azimuth
                transformed_frame = self.create_centered_360_view(frame, center_azimuth, 
                                                                output_size[0], output_size[1])
                frames.append(transformed_frame)
            else:
                break
        
        cap.release()
        return frames
    
    def save_video_clip(self, frames: List[np.ndarray], output_path: str, fps: float = 30.0):
        """Save frames as video clip"""
        if not frames:
            raise ValueError("No frames to save")
            
        temp_path = output_path.replace('.mp4', '_tmp.mp4')
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 改为 H264 编码
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
            
        out.release()
        
        # Re-encode to H.264 using ffmpeg
        cmd = [
            "ffmpeg", "-y", "-i", temp_path,
            "-c:v", "libx264", "-crf", "23", "-preset", "medium",
            "-an",  # 音频用AAC（如果没有音频轨道也没问题）
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=False)

        # 删除临时文件
        os.remove(temp_path)
    
    def get_representative_frame(self, frames: List[np.ndarray]) -> np.ndarray:
        """Get a representative frame from video clip (middle frame)"""
        if not frames:
            raise ValueError("No frames available")
        return frames[len(frames) // 2]
    
    def generate_video_choice_question(self, audio_foa_path: str, audio_mic_path: str, 
                                     video_path: str, event: pd.Series) -> Dict[str, Any]:
        """Generate a question where user gets audio and chooses video (original type)"""
        
        # Calculate timestamp from frame number
        timestamp = event['frame'] / self.metadata_fps
        
        # Generate video clips for different viewing directions
        clip_duration = 5.0  # 2 seconds
        
        # Load audio segment (2 seconds around the event)
        # audio_foa, sr = self.load_audio(audio_foa_path)
        audio_mic, sr = self.load_audio(audio_mic_path)
        audio_segment = self.extract_audio_segment(audio_mic, timestamp, clip_duration, sr)
        
        # Generate video clips for different viewing directions
        correct_azimuth = 0
        correct_clip = self.extract_video_segment(video_path, timestamp, clip_duration, correct_azimuth)
        
        # Generate 3 incorrect views with different center azimuths
        incorrect_azimuths = [90, 180, 270]
        incorrect_clips = []
        for azimuth in incorrect_azimuths:
            clip = self.extract_video_segment(video_path, timestamp, clip_duration, azimuth)
            incorrect_clips.append(clip)
        
        # Combine all clips and shuffle
        all_clips = [correct_clip] + incorrect_clips
        all_center_azimuths = [correct_azimuth] + incorrect_azimuths
        
        # Create random order
        order = list(range(4))
        random.shuffle(order)
        
        shuffled_clips = [all_clips[i] for i in order]
        shuffled_azimuths = [all_center_azimuths[i] for i in order]
        correct_answer = order.index(0)  # Find where the correct clip ended up
        
        # Get representative frames for visualization
        representative_frames = [self.get_representative_frame(clip) for clip in shuffled_clips]
        return {
            'question_type': 'video_choice',
            'question': f"In the given 360° panoramic video, several candidate views are shown, each representing a different viewing direction. The forward direction of each view corresponds to the middle of the image. You will also be provided with a spatial audio clip in which a {event['class_name'].lower()} sound can be heard from a specific direction within the same 3D space. Which candidate video view has its viewing direction aligned with the direction from which the sound is coming?",
            'input_audio': audio_segment,
            'input_audio_path': audio_foa_path,
            'sample_rate': sr,
            'choice_video_clips': shuffled_clips,
            'choice_video_paths': [video_path] * 4,  # Same video, different rotations
            'choice_video_rotations': shuffled_azimuths,
            'representative_frames': representative_frames,
            'correct_answer': correct_answer,
            'event_info': {
                'class': event['class_name'],
                'timestamp': timestamp,
                'azimuth': event['azimuth'],
                'elevation': event['elevation'],
                'distance': event['distance'],
                'original_azimuth': event['azimuth'],
                'correct_center_azimuth': correct_azimuth
            }
        }
    
    def generate_audio_choice_question(self, audio_foa_path: str, audio_mic_path: str, 
                                     video_path: str, event: pd.Series) -> Dict[str, Any]:
        """Generate a question where user gets video and chooses audio (new type)"""
        
        # Calculate timestamp from frame number
        timestamp = event['frame'] / self.metadata_fps
        
        clip_duration = 5.0  # 2 seconds
        # Load audio segment (2 seconds around the event)
        # audio_foa, sr = self.load_audio(audio_foa_path)
        audio_mic, sr = self.load_audio(audio_mic_path)
        audio_segment = self.extract_audio_segment(audio_mic, timestamp, clip_duration, sr)
        
        # Generate input video with view angle 0 (no rotation)
        input_video_clip = self.extract_video_segment(video_path, timestamp, clip_duration, 0)
        
        # Generate correct audio (no rotation)
        correct_rotation = 0
        correct_audio = audio_segment.copy()
        
        # Generate 3 incorrect audio clips with different rotations
        incorrect_rotations = [90, 180, 270]
        incorrect_audios = []
        for rotation in incorrect_rotations:
            # rotated_audio = self.rotate_audio_azimuth(audio_segment, rotation)
            rotated_audio = self.rotate_audio_azimuth_mic(audio_segment, rotation)
            incorrect_audios.append(rotated_audio)
        
        # Combine all audio clips and shuffle
        all_audios = [correct_audio] + incorrect_audios
        all_rotations = [correct_rotation] + incorrect_rotations
        
        # Create random order
        order = list(range(4))
        random.shuffle(order)
        
        shuffled_audios = [all_audios[i] for i in order]
        shuffled_rotations = [all_rotations[i] for i in order]
        correct_answer = order.index(0)  # Find where the correct audio ended up
        
        return {
            'question_type': 'audio_choice',
            'question': f"You are provided with a 360° panoramic video showing the scene from a specific viewing direction (forward direction is in the middle of the image). You will also hear several spatial audio clips, each representing the same {event['class_name'].lower()} sound but from different listening orientations. Which audio clip matches the spatial perspective of the given video view?",
            'input_video_clip': input_video_clip,
            'input_video_path': video_path,
            'input_video_rotation': 0,  # Always view angle 0 for input
            'choice_audios': shuffled_audios,
            'choice_audio_paths': [audio_foa_path] * 4,  # Same audio, different rotations
            'choice_audio_rotations': shuffled_rotations,
            'sample_rate': sr,
            'correct_answer': correct_answer,
            'representative_frame': self.get_representative_frame(input_video_clip),
            'event_info': {
                'class': event['class_name'],
                'timestamp': timestamp,
                'azimuth': event['azimuth'],
                'elevation': event['elevation'],
                'distance': event['distance'],
                'original_azimuth': event['azimuth']
            }
        }
    
    def visualize_video_choice_question(self, question_data: Dict[str, Any], save_path: str = None):
        """Visualize the video choice question using only one representative frame"""
        # Use the correct answer's representative frame for visualization
        correct_idx = question_data['correct_answer']
        representative_frame = question_data['representative_frames'][correct_idx]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Convert BGR to RGB for matplotlib
        frame_rgb = cv2.cvtColor(representative_frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)
        
        # Add title and information
        center_azimuth = question_data['choice_video_rotations'][correct_idx]
        ax.set_title(f"Representative Frame - Correct View (Center: {center_azimuth}°)", 
                    color='green', fontweight='bold', fontsize=14)
        ax.axis('off')
        
        # Add a vertical line in the middle to show the center direction
        height, width = representative_frame.shape[:2]
        ax.axvline(x=width//2, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add text information
        event_info = question_data['event_info']
        info_text = (f"Event: {event_info['class']} at {event_info['timestamp']:.1f}s\n"
                    f"Ground Truth Azimuth: {event_info['azimuth']}°\n"
                    f"Elevation: {event_info['elevation']}°, Distance: {event_info['distance']}")
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=10)
        
        plt.suptitle(f"Audio-Visual Localization Question (Video Choice)\n{question_data['question'][:100]}...", 
                    fontsize=11, y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def visualize_audio_choice_question(self, question_data: Dict[str, Any], save_path: str = None):
        """Visualize the audio choice question using the input video frame"""
        representative_frame = question_data['representative_frame']
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Convert BGR to RGB for matplotlib
        frame_rgb = cv2.cvtColor(representative_frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)
        
        # Add title and information
        ax.set_title(f"Input Video Frame (View Angle: 0°)", 
                    color='blue', fontweight='bold', fontsize=14)
        ax.axis('off')
        
        # Add a vertical line in the middle to show the center direction
        height, width = representative_frame.shape[:2]
        ax.axvline(x=width//2, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add text information
        event_info = question_data['event_info']
        info_text = (f"Event: {event_info['class']} at {event_info['timestamp']:.1f}s\n"
                    f"Ground Truth Azimuth: {event_info['azimuth']}°\n"
                    f"Elevation: {event_info['elevation']}°, Distance: {event_info['distance']}\n"
                    f"Audio rotations: {question_data['choice_audio_rotations']}")
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=10)
        
        plt.suptitle(f"Audio-Visual Localization Question (Audio Choice)\n{question_data['question'][:100]}...", 
                    fontsize=11, y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def save_video_choice_question(self, question_data, output_dir: str, question_id: str):
        """Save video choice question data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save input audio
        audio = np.asarray(question_data['input_audio'])
        sr = int(question_data['sample_rate'])

        # Ensure proper audio format
        if audio.ndim == 1:
            pass  # mono already
        else:
            if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
                audio = audio.T

        if np.issubdtype(audio.dtype, np.floating):
            audio = np.clip(audio, -1.0, 1.0)

        audio_file = output_path / f"{question_id}_input_audio.wav"
        sf.write(str(audio_file), audio, sr, subtype="PCM_16")

        # Save choice video clips
        for i, video_clip in enumerate(question_data['choice_video_clips']):
            video_file = output_path / f"{question_id}_choice_{i}.mp4"
            try:
                self.save_video_clip(video_clip, str(video_file), fps=30.0)
            except Exception as e:
                print(f"Failed to save video clip {i}: {e}")
                raise
        
        # Save visualization
        viz_file = output_path / f"{question_id}_visualization.png"
        self.visualize_video_choice_question(question_data, str(viz_file))
        
        # Save metadata
        metadata = {
            'question_type': question_data['question_type'],
            'question': question_data['question'],
            'correct_answer': int(question_data['correct_answer']),
            'input_audio_path': question_data['input_audio_path'],
            'choice_video_paths': question_data['choice_video_paths'],
            'choice_video_rotations': [int(x) for x in question_data['choice_video_rotations']],
            'event_info': {
                'class': question_data['event_info']['class'],
                'timestamp': float(question_data['event_info']['timestamp']),
                'azimuth': int(question_data['event_info']['azimuth']),
                'elevation': int(question_data['event_info']['elevation']),
                'distance': int(question_data['event_info']['distance']),
                'original_azimuth': int(question_data['event_info']['original_azimuth']),
                'correct_center_azimuth': int(question_data['event_info']['correct_center_azimuth'])
            }
        }
        
        with open(output_path / f"{question_id}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def save_audio_choice_question(self, question_data, output_dir: str, question_id: str):
        """Save audio choice question data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save input video
        video_file = output_path / f"{question_id}_input_video.mp4"
        try:
            self.save_video_clip(question_data['input_video_clip'], str(video_file), fps=30.0)
        except Exception as e:
            print(f"Failed to save input video: {e}")
            raise

        # Save choice audio clips
        for i, audio_clip in enumerate(question_data['choice_audios']):
            audio = np.asarray(audio_clip)
            sr = int(question_data['sample_rate'])

            # Ensure proper audio format
            if audio.ndim == 1:
                pass  # mono already
            else:
                if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
                    audio = audio.T

            if np.issubdtype(audio.dtype, np.floating):
                audio = np.clip(audio, -1.0, 1.0)

            audio_file = output_path / f"{question_id}_choice_{i}.wav"
            sf.write(str(audio_file), audio, sr, subtype="PCM_16")
        
        # Save visualization
        viz_file = output_path / f"{question_id}_visualization.png"
        self.visualize_audio_choice_question(question_data, str(viz_file))
        
        # Save metadata
        metadata = {
            'question_type': question_data['question_type'],
            'question': question_data['question'],
            'correct_answer': int(question_data['correct_answer']),
            'input_video_path': question_data['input_video_path'],
            'input_video_rotation': int(question_data['input_video_rotation']),
            'choice_audio_paths': question_data['choice_audio_paths'],
            'choice_audio_rotations': [int(x) for x in question_data['choice_audio_rotations']],
            'event_info': {
                'class': question_data['event_info']['class'],
                'timestamp': float(question_data['event_info']['timestamp']),
                'azimuth': int(question_data['event_info']['azimuth']),
                'elevation': int(question_data['event_info']['elevation']),
                'distance': int(question_data['event_info']['distance']),
                'original_azimuth': int(question_data['event_info']['original_azimuth'])
            }
        }
        
        with open(output_path / f"{question_id}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


def process_all_clips_with_skip():
    """Process all clips and generate both types of questions for all audio events, skipping processed ones"""
    generator = STARSS23QuestionGenerator()
    
    # Base paths
    base_path = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/STARSS23"
    base_output_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/STARSS23_processed"
    
    # Define all splits and their paths
    splits = [
        "dev-test-sony",
        # "dev-train-sony", 
        "dev-test-tau",
        # "dev-train-tau"
    ]
    
    total_video_choice_questions = 0
    total_audio_choice_questions = 0
    skipped_video_choice = 0
    skipped_audio_choice = 0
    failed_clips = []
    
    for split in splits:
        print(f"\n=== Processing {split} ===")
        
        # Paths for this split
        foa_dir = f"{base_path}/foa_dev/{split}"
        mic_dir = f"{base_path}/mic_dev/{split}"
        video_dir = f"{base_path}/video_dev/{split}"
        metadata_dir = f"{base_path}/metadata_dev/{split}"
        
        # Get all wav files in this split
        wav_files = glob.glob(f"{foa_dir}/*.wav")
        
        for wav_file in tqdm(wav_files, desc=f"Processing {split}"):
            # Extract filename without extension
            filename = Path(wav_file).stem
            
            # Construct file paths
            audio_foa_path = f"{foa_dir}/{filename}.wav"
            audio_mic_path = f"{mic_dir}/{filename}.wav"
            video_path = f"{video_dir}/{filename}.mp4"
            metadata_path = f"{metadata_dir}/{filename}.csv"
            
            # Check if all files exist
            if not all(Path(p).exists() for p in [audio_foa_path, audio_mic_path, video_path, metadata_path]):
                continue

            try:
                # Load metadata to get all events
                metadata = generator.load_metadata(metadata_path)
                
                if len(metadata) == 0:
                    continue
                
                # Remove duplicates (same frame, class, azimuth)
                unique_events = metadata.drop_duplicates(subset=['frame', 'class_idx', 'azimuth'])
                
                # Process each unique event (limit to avoid too many questions per clip)
                max_events_per_clip = 5
                if len(unique_events) > max_events_per_clip:
                    unique_events = unique_events.sample(n=max_events_per_clip, random_state=42)
                
                # Process each unique event for both question types
                for event_idx, event in unique_events.iterrows():
                    # Generate question IDs
                    base_question_id = f"event_{event['frame']:04d}_{event['class_idx']:02d}_az{event['azimuth']:03d}"
                    video_choice_id = f"{base_question_id}_video_choice"
                    audio_choice_id = f"{base_question_id}_audio_choice"
                    
                    # Define output directories
                    video_choice_output_dir = f"{base_output_dir}/questions_video_choice_v2/{split}/{filename}"
                    audio_choice_output_dir = f"{base_output_dir}/questions_audio_choice_v2/{split}/{filename}"
                    
                    # Check if video choice question already processed
                    video_choice_processed = generator.check_event_processed(
                        video_choice_output_dir, video_choice_id, 'video_choice'
                    )
                    
                    # Check if audio choice question already processed  
                    audio_choice_processed = generator.check_event_processed(
                        audio_choice_output_dir, audio_choice_id, 'audio_choice'
                    )
                    
                    # Process video choice question if not already done
                    if not video_choice_processed:
                        video_choice_question = generator.generate_video_choice_question(
                            audio_foa_path, audio_mic_path, video_path, event
                        )
                        generator.save_video_choice_question(video_choice_question, video_choice_output_dir, video_choice_id)
                        total_video_choice_questions += 1
                    else:
                        skipped_video_choice += 1
                    
                    # Process audio choice question if not already done
                    if not audio_choice_processed:
                        audio_choice_question = generator.generate_audio_choice_question(
                            audio_foa_path, audio_mic_path, video_path, event
                        )
                        generator.save_audio_choice_question(audio_choice_question, audio_choice_output_dir, audio_choice_id)
                        total_audio_choice_questions += 1
                    else:
                        skipped_audio_choice += 1
                        
            except Exception as e:
                failed_clips.append(f"{split}/{filename}: {str(e)}")
                continue
    
    print(f"\n=== Summary ===")
    print(f"Total video choice questions generated: {total_video_choice_questions}")
    print(f"Total audio choice questions generated: {total_audio_choice_questions}")
    print(f"Skipped video choice questions (already processed): {skipped_video_choice}")
    print(f"Skipped audio choice questions (already processed): {skipped_audio_choice}")
    print(f"Total questions generated: {total_video_choice_questions + total_audio_choice_questions}")
    print(f"Total questions skipped: {skipped_video_choice + skipped_audio_choice}")
    print(f"Failed clips: {len(failed_clips)}")
    if failed_clips:
        print("Failed clips:")
        for clip in failed_clips:
            print(f"  - {clip}")


def process_single_event_with_skip(args):
    """
    Process a single event with skip functionality - for parallel processing
    """
    (audio_foa_path, audio_mic_path, video_path, metadata_path, 
     split, filename, event_data, base_output_dir) = args
    
    try:
        # Create generator instance (each process needs its own)
        generator = STARSS23QuestionGenerator()
        
        # Convert event_data back to Series
        event = pd.Series(event_data)
        
        # Generate question IDs
        base_question_id = f"event_{event['frame']:04d}_{event['class_idx']:02d}_az{event['azimuth']:03d}"
        video_choice_id = f"{base_question_id}_video_choice"
        audio_choice_id = f"{base_question_id}_audio_choice"
        
        # Define output directories
        video_choice_output_dir = f"{base_output_dir}/questions_video_choice_v2/{split}/{filename}"
        audio_choice_output_dir = f"{base_output_dir}/questions_audio_choice_v2/{split}/{filename}"
        
        # Check if questions already processed
        video_choice_processed = generator.check_event_processed(
            video_choice_output_dir, video_choice_id, 'video_choice'
        )
        audio_choice_processed = generator.check_event_processed(
            audio_choice_output_dir, audio_choice_id, 'audio_choice'
        )
        
        results = {
            'success': True,
            'event_class': event['class_name'],
            'azimuth': event['azimuth'],
            'split': split,
            'filename': filename,
            'video_choice_generated': 0,
            'audio_choice_generated': 0,
            'video_choice_skipped': 0,
            'audio_choice_skipped': 0
        }
        
        # Process video choice question if not already done
        if not video_choice_processed:
            video_choice_question = generator.generate_video_choice_question(
                audio_foa_path, audio_mic_path, video_path, event
            )
            generator.save_video_choice_question(video_choice_question, video_choice_output_dir, video_choice_id)
            results['video_choice_generated'] = 1
        else:
            results['video_choice_skipped'] = 1
        
        # Process audio choice question if not already done
        if not audio_choice_processed:
            audio_choice_question = generator.generate_audio_choice_question(
                audio_foa_path, audio_mic_path, video_path, event
            )
            generator.save_audio_choice_question(audio_choice_question, audio_choice_output_dir, audio_choice_id)
            results['audio_choice_generated'] = 1
        else:
            results['audio_choice_skipped'] = 1
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'split': split,
            'filename': filename,
            'event_data': event_data,
            'video_choice_generated': 0,
            'audio_choice_generated': 0,
            'video_choice_skipped': 0,
            'audio_choice_skipped': 0
        }


def process_all_clips_parallel_with_skip(max_workers=None):
    """
    并行处理所有剪辑，跳过已处理的事件 - 以事件为单位分配任务
    """
    # Base paths
    base_path = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/STARSS23"
    base_output_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/STARSS23_processed"
    
    # Define all splits and their paths
    splits = [
        "dev-test-sony",
        "dev-test-tau",
    ]
    
    # Collect all events to process
    all_events = []
    generator = STARSS23QuestionGenerator()  # For metadata loading only
    
    print("Collecting all events to process...")
    for split in splits:
        print(f"Processing {split}...")
        
        foa_dir = f"{base_path}/foa_dev/{split}"
        mic_dir = f"{base_path}/mic_dev/{split}"
        video_dir = f"{base_path}/video_dev/{split}"
        metadata_dir = f"{base_path}/metadata_dev/{split}"
        
        wav_files = glob.glob(f"{foa_dir}/*.wav")
        
        for wav_file in tqdm(wav_files, desc=f"Loading {split}"):
            filename = Path(wav_file).stem
            
            # Construct file paths
            audio_foa_path = f"{foa_dir}/{filename}.wav"
            audio_mic_path = f"{mic_dir}/{filename}.wav"
            video_path = f"{video_dir}/{filename}.mp4"
            metadata_path = f"{metadata_dir}/{filename}.csv"
            
            # Check if all files exist
            if not all(Path(p).exists() for p in [audio_foa_path, audio_mic_path, video_path, metadata_path]):
                continue
            
            try:
                # Load metadata to get all events
                metadata = generator.load_metadata(metadata_path)
                
                if len(metadata) == 0:
                    continue
                
                # Remove duplicates (same frame, class, azimuth)
                unique_events = metadata.drop_duplicates(subset=['frame', 'class_idx', 'azimuth'])
                
                # Process each unique event (limit to avoid too many questions per clip)
                max_events_per_clip = 5
                if len(unique_events) > max_events_per_clip:
                    unique_events = unique_events.sample(n=max_events_per_clip, random_state=42)
                
                # Add each event as a separate task
                for event_idx, event in unique_events.iterrows():
                    event_args = (
                        audio_foa_path, audio_mic_path, video_path, metadata_path,
                        split, filename, event.to_dict(), base_output_dir
                    )
                    all_events.append(event_args)
                    
            except Exception as e:
                print(f"Failed to load metadata for {split}/{filename}: {e}")
                continue
    
    print(f"Found {len(all_events)} events to process")
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(mp.cpu_count() - 1, len(all_events))  # Leave one CPU free
    
    print(f"Using {max_workers} worker processes")
    
    # Process events in parallel
    total_video_choice_questions = 0
    total_audio_choice_questions = 0
    skipped_video_choice = 0
    skipped_audio_choice = 0
    failed_events = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_event = {
            executor.submit(process_single_event_with_skip, event_args): event_args 
            for event_args in all_events
        }
        
        # Process results with progress bar
        with tqdm(total=len(all_events), desc="Processing events") as pbar:
            for future in as_completed(future_to_event):
                event_args = future_to_event[future]
                split, filename = event_args[4], event_args[5]
                
                try:
                    result = future.result()
                    
                    if result['success']:
                        total_video_choice_questions += result['video_choice_generated']
                        total_audio_choice_questions += result['audio_choice_generated']
                        skipped_video_choice += result['video_choice_skipped']
                        skipped_audio_choice += result['audio_choice_skipped']
                        
                        # Show status
                        status_parts = []
                        if result['video_choice_generated'] or result['audio_choice_generated']:
                            status_parts.append("Generated")
                        if result['video_choice_skipped'] or result['audio_choice_skipped']:
                            status_parts.append("Skipped")
                        
                        pbar.set_postfix({
                            'Status': '/'.join(status_parts) if status_parts else 'Processed',
                            'Event': f"{result['event_class']} @ {result['azimuth']}°",
                            'File': f"{split[:3]}/{filename[:10]}"
                        })
                    else:
                        failed_events.append(f"{split}/{filename}: {result['error']}")
                        pbar.set_postfix({
                            'Failed': f"{split}/{filename}",
                            'Error': result['error'][:30]
                        })
                
                except Exception as e:
                    failed_events.append(f"{split}/{filename}: {str(e)}")
                    pbar.set_postfix({
                        'Error': f"{split}/{filename}",
                        'Exception': str(e)[:30]
                    })
                
                pbar.update(1)
    
    print(f"\n=== Summary ===")
    print(f"Total video choice questions generated: {total_video_choice_questions}")
    print(f"Total audio choice questions generated: {total_audio_choice_questions}")
    print(f"Skipped video choice questions (already processed): {skipped_video_choice}")
    print(f"Skipped audio choice questions (already processed): {skipped_audio_choice}")
    print(f"Total questions generated: {total_video_choice_questions + total_audio_choice_questions}")
    print(f"Total questions skipped: {skipped_video_choice + skipped_audio_choice}")
    print(f"Failed events: {len(failed_events)}")
    if failed_events:
        print("Failed events:")
        for event in failed_events[:10]:  # Show first 10 failures
            print(f"  - {event}")
        if len(failed_events) > 10:
            print(f"  ... and {len(failed_events) - 10} more")


def main():
    """
    主函数 - 提供不同的处理选项
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='STARSS23 Question Generator with Skip Functionality')
    parser.add_argument('--mode', choices=['serial', 'parallel'], 
                       default='parallel',
                       help='Processing mode: serial (single-thread), parallel (multi-process)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker processes for parallel mode (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    print(f"Running in {args.mode} mode with skip functionality")
    if args.mode == 'parallel':
        print(f"Using {args.workers or 'auto'} workers")
    
    start_time = time.time()
    
    if args.mode == 'serial':
        process_all_clips_with_skip()
    elif args.mode == 'parallel':
        process_all_clips_parallel_with_skip(max_workers=args.workers)
    
    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()