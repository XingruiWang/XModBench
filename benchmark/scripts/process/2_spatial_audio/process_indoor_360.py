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
    
    def create_perspective_view(self, frame: np.ndarray, yaw: float, pitch: float = 0, 
                              fov: float = 90, output_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        """Convert 360° equirectangular frame to perspective view"""
        height, width = frame.shape[:2]
        output_width, output_height = output_size
        
        # Create coordinate mapping
        x_map = np.zeros((output_height, output_width), dtype=np.float32)
        y_map = np.zeros((output_height, output_width), dtype=np.float32)
        
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        fov_rad = math.radians(fov)
        
        for y in range(output_height):
            for x in range(output_width):
                # Normalize coordinates
                norm_x = (2.0 * x / output_width) - 1.0
                norm_y = (2.0 * y / output_height) - 1.0
                
                # Project to 3D sphere
                z = 1.0
                x_3d = norm_x * math.tan(fov_rad / 2)
                y_3d = norm_y * math.tan(fov_rad / 2)
                
                # Normalize
                length = math.sqrt(x_3d*x_3d + y_3d*y_3d + z*z)
                x_3d /= length
                y_3d /= length
                z /= length
                
                # Apply rotations
                x_rot = x_3d * math.cos(yaw_rad) + z * math.sin(yaw_rad)
                z_rot = -x_3d * math.sin(yaw_rad) + z * math.cos(yaw_rad)
                
                y_rot = y_3d * math.cos(pitch_rad) - z_rot * math.sin(pitch_rad)
                z_final = y_3d * math.sin(pitch_rad) + z_rot * math.cos(pitch_rad)
                
                # Convert to equirectangular coordinates
                longitude = math.atan2(x_rot, z_final)
                latitude = math.asin(max(-1, min(1, y_rot)))
                
                # Map to image coordinates
                u = (longitude + math.pi) / (2 * math.pi) * width
                v = (math.pi/2 - latitude) / math.pi * height
                
                x_map[y, x] = u
                y_map[y, x] = v
        
        # Remap the frame
        perspective_frame = cv2.remap(frame, x_map, y_map, cv2.INTER_LINEAR)
        
        # Fix orientation - flip vertically
        perspective_frame = cv2.flip(perspective_frame, 0)  # Flip vertically
        
        return perspective_frame
    
    def get_video_frame_at_time(self, video_path: str, timestamp: float) -> np.ndarray:
        """Extract a frame from 360° video at specific timestamp"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame number
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        else:
            raise ValueError(f"Could not read frame at timestamp {timestamp}")
    
    def filter_events_by_direction(self, metadata: pd.DataFrame, target_per_direction: int = 5) -> pd.DataFrame:
        """Filter events to get balanced distribution across 4 cardinal directions"""
        
        def get_direction_from_azimuth(azimuth):
            azimuth = azimuth % 360  # Normalize to 0-360
            
            # Check North (wraps around 0°)
            if azimuth >= 315 or azimuth < 45:
                return 0
            elif 45 <= azimuth < 135:
                return 90
            elif 135 <= azimuth < 225:
                return 180
            elif 225 <= azimuth < 315:
                return 270
            else:
                return 0  # Default to North
        
        # Add direction column
        metadata['direction'] = metadata['azimuth'].apply(get_direction_from_azimuth)
        
        # Remove duplicates (same frame, class, direction)
        unique_events = metadata.drop_duplicates(subset=['frame', 'class_idx', 'direction'])
        
        # Select events for each direction
        selected_events = []
        
        for direction in [0, 90, 180, 270]:
            direction_events = unique_events[unique_events['direction'] == direction]
            
            if len(direction_events) > 0:
                # Sample up to target_per_direction events from this direction
                n_samples = min(target_per_direction, len(direction_events))
                sampled = direction_events.sample(n=n_samples, random_state=42)
                selected_events.append(sampled)
                
                print(f"  Direction {direction}°: selected {n_samples} events (out of {len(direction_events)} available)")
            else:
                print(f"  Direction {direction}°: no events found")
        
        if selected_events:
            filtered_metadata = pd.concat(selected_events, ignore_index=True)
            return filtered_metadata
        else:
            return pd.DataFrame()  # Return empty if no events found
    
    def generate_question_for_event(self, audio_foa_path: str, audio_mic_path: str, 
                                  video_path: str, event: pd.Series) -> Dict[str, Any]:
        """Generate a question for a specific event"""
        
        # Calculate timestamp from frame number
        timestamp = event['frame'] / self.metadata_fps
        
        # Load audio segment (2 seconds around the event)
        audio_foa, sr = self.load_audio(audio_foa_path)
        audio_segment = self.extract_audio_segment(audio_foa, max(0, timestamp - 1), 2.0, sr)
        
        # Get video frame
        video_frame = self.get_video_frame_at_time(video_path, timestamp)
        
        # Use 4 cardinal directions: North(0°), East(90°), South(180°), West(270°)
        all_azimuths = [0, 90, 180, 270]
        
        # Generate views for all 4 directions
        all_views = []
        for azimuth in all_azimuths:
            view = self.create_perspective_view(video_frame, azimuth)
            all_views.append(view)
        
        # The correct answer is determined by the event's direction
        # This was already calculated in filter_events_by_direction
        correct_answer = all_azimuths.index(event['direction'])
        
        return {
            'question': f"Which image shows the correct view direction for the {event['class_name'].lower()} sound?",
            'audio_segment': audio_segment,
            'sample_rate': sr,
            'views': all_views,
            'view_azimuths': all_azimuths,
            'correct_answer': correct_answer,
            'event_info': {
                'class': event['class_name'],
                'timestamp': timestamp,
                'azimuth': event['azimuth'],
                'elevation': event['elevation'],
                'distance': event['distance'],
                'original_azimuth': event['azimuth'],
                'mapped_direction': event['direction'],
                'correct_answer_direction': all_azimuths[correct_answer]
            }
        }
    
    def visualize_question(self, question_data: Dict[str, Any], save_path: str = None):
        """Visualize the generated question and save to file"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for i, (ax, view) in enumerate(zip(axes.flat, question_data['views'])):
            # Convert BGR to RGB for matplotlib
            view_rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
            ax.imshow(view_rgb)
            
            title = f"Option {i+1} (Azimuth: {question_data['view_azimuths'][i]}°)"
            if i == question_data['correct_answer']:
                title += " ✓ CORRECT"
                ax.set_title(title, color='green', fontweight='bold')
            else:
                ax.set_title(title)
            ax.axis('off')
        
        plt.suptitle(f"Question: {question_data['question']}\n"
                    f"Event: {question_data['event_info']['class']} at "
                    f"{question_data['event_info']['timestamp']:.1f}s", fontsize=14)
        plt.tight_layout()
        
        # Save to file instead of showing
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.savefig('question_visualization.png', dpi=150, bbox_inches='tight')
            print("Visualization saved to: question_visualization.png")
        
        plt.close()  # Close the figure to free memory
    
    def save_question(self, question_data, output_dir: str, question_id: str):
        """Save question data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # ---- Save audio (expects float in [-1, 1]) ----
        audio = np.asarray(question_data['audio_segment'])
        sr = int(question_data['sample_rate'])

        # Ensure shape is (n_samples, n_channels)
        if audio.ndim == 1:
            pass  # mono already
        else:
            # Common case: (channels, samples) -> transpose to (samples, channels)
            if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
                audio = audio.T
            # If it's already (samples, channels), do nothing

        # Optional: clamp to [-1, 1] if upstream might exceed
        if np.issubdtype(audio.dtype, np.floating):
            audio = np.clip(audio, -1.0, 1.0)

        audio_file = output_path / f"{question_id}_audio.wav"
        sf.write(str(audio_file), audio, sr, subtype="PCM_16")

        # ---- Save images ----
        for i, view in enumerate(question_data['views']):
            img = view
            # If your images are RGB (e.g., from PIL/matplotlib), convert to BGR for OpenCV
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_file = output_path / f"{question_id}_option_{i}.jpg"
            ok = cv2.imwrite(str(img_file), img)
            if not ok:
                raise IOError(f"Failed to write image: {img_file}")
        
        # Save visualization
        viz_file = output_path / f"{question_id}_visualization.png"
        self.visualize_question(question_data, str(viz_file))
        
        # Save metadata - Convert numpy types to Python native types for JSON serialization
        metadata = {
            'question': question_data['question'],
            'correct_answer': int(question_data['correct_answer']),  # Convert to int
            'view_azimuths': [int(x) for x in question_data['view_azimuths']],  # Convert list elements
            'event_info': {
                'class': question_data['event_info']['class'],
                'timestamp': float(question_data['event_info']['timestamp']),
                'azimuth': int(question_data['event_info']['azimuth']),
                'elevation': int(question_data['event_info']['elevation']),
                'distance': int(question_data['event_info']['distance']),
                'original_azimuth': int(question_data['event_info']['original_azimuth']),
                'mapped_direction': int(question_data['event_info']['mapped_direction']),
                'correct_answer_direction': int(question_data['event_info']['correct_answer_direction'])
            }
        }
        
        with open(output_path / f"{question_id}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


def process_all_clips():
    """Process all clips and generate questions for all audio events"""
    generator = STARSS23QuestionGenerator()
    
    # Base paths
    base_path = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/STARSS23"
    output_base = "output/questions_all"
    
    # Define all splits and their paths
    splits = [
        "dev-test-sony",
        "dev-train-sony", 
        "dev-test-tau",
        "dev-train-tau"
    ]
    
    total_questions = 0
    failed_clips = []
    direction_counts = {0: 0, 90: 0, 180: 0, 270: 0}  # Track distribution
    
    for split in splits:
        print(f"\n=== Processing {split} ===")
        
        # Paths for this split
        foa_dir = f"{base_path}/foa_dev/{split}"
        mic_dir = f"{base_path}/mic_dev/{split}"
        video_dir = f"{base_path}/video_dev/{split}"
        metadata_dir = f"{base_path}/metadata_dev/{split}"
        
        # Get all wav files in this split
        wav_files = glob.glob(f"{foa_dir}/*.wav")
        
        for wav_file in wav_files:
            # Extract filename without extension
            filename = Path(wav_file).stem
            print(f"Processing {filename}...")
            
            # Construct file paths
            audio_foa_path = f"{foa_dir}/{filename}.wav"
            audio_mic_path = f"{mic_dir}/{filename}.wav"
            video_path = f"{video_dir}/{filename}.mp4"
            metadata_path = f"{metadata_dir}/{filename}.csv"
            
            # Check if all files exist
            if not all(Path(p).exists() for p in [audio_foa_path, audio_mic_path, video_path, metadata_path]):
                print(f"  Skipping {filename} - missing files")
                continue
            
            try:
                # Load metadata to get all events
                metadata = generator.load_metadata(metadata_path)
                
                if len(metadata) == 0:
                    print(f"  Skipping {filename} - no events found")
                    continue
                
                # Filter events to get balanced distribution across 4 directions
                print(f"  Original events: {len(metadata)}")
                filtered_events = generator.filter_events_by_direction(metadata, target_per_direction=3)
                
                if len(filtered_events) == 0:
                    print(f"  Skipping {filename} - no events after filtering")
                    continue
                
                print(f"  Filtered events: {len(filtered_events)} (balanced across directions)")
                
                # Process each filtered event
                for event_idx, event in filtered_events.iterrows():
                    try:
                        # Generate question for this specific event
                        question_data = generator.generate_question_for_event(
                            audio_foa_path, audio_mic_path, video_path, event
                        )
                        
                        # Create output folder for this clip
                        clip_output_dir = f"{output_base}/{split}/{filename}"
                        
                        # Generate question ID
                        question_id = f"event_{event['frame']:04d}_{event['class_idx']:02d}_dir{event['direction']:03d}"
                        
                        # Save question
                        generator.save_question(question_data, clip_output_dir, question_id)
                        
                        total_questions += 1
                        direction_counts[event['direction']] += 1  # Track distribution
                        
                        print(f"    Generated question {question_id}: {event['class_name']} at {event['azimuth']}° -> direction {event['direction']}°")
                        
                    except Exception as e:
                        print(f"    Failed to generate question for event {event_idx}: {e}")
                        continue
                        
            except Exception as e:
                print(f"  Failed to process {filename}: {e}")
                failed_clips.append(f"{split}/{filename}")
                continue
    
    print(f"\n=== Summary ===")
    print(f"Total questions generated: {total_questions}")
    print(f"Direction distribution:")
    for direction, count in direction_counts.items():
        percentage = (count / total_questions * 100) if total_questions > 0 else 0
        direction_name = {0: "North", 90: "East", 180: "South", 270: "West"}[direction]
        print(f"  {direction}° ({direction_name}): {count} questions ({percentage:.1f}%)")
    print(f"Failed clips: {len(failed_clips)}")
    if failed_clips:
        print("Failed clips:")
        for clip in failed_clips:
            print(f"  - {clip}")


def main():
    # Initialize the question generator and process all clips
    process_all_clips()


if __name__ == "__main__":
    main()