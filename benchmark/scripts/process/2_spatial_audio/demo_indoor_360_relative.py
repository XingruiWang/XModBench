import cv2
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import math
import random
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
        
        # Fix orientation - flip vertically if needed
        # You may need to adjust this based on your specific 360° camera orientation
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
    
    def generate_question_data(self, audio_foa_path: str, audio_mic_path: str, 
                             video_path: str, metadata_path: str) -> Dict[str, Any]:
        """Generate a complete question with audio-visual matching task"""
        
        # Load metadata
        metadata = self.load_metadata(metadata_path)
        
        if len(metadata) == 0:
            raise ValueError("No valid metadata found")
        
        # Select a random event
        event = metadata.sample(1).iloc[0]
        
        # Calculate timestamp from frame number
        timestamp = event['frame'] / self.metadata_fps
        
        # Load audio segment (2 seconds around the event)
        audio_foa, sr = self.load_audio(audio_foa_path)
        audio_segment = self.extract_audio_segment(audio_foa, max(0, timestamp - 1), 2.0, sr)
        
        # Get video frame
        video_frame = self.get_video_frame_at_time(video_path, timestamp)
        
        # Generate correct perspective view (where sound is coming from)
        correct_azimuth = event['azimuth'] % 360
        correct_view = self.create_perspective_view(video_frame, correct_azimuth)
        
        # Generate 3 incorrect views (different azimuths)
        incorrect_azimuths = []
        for n in range(3):
            incorrect_azimuth = (correct_azimuth + n*90 + 90) % 360
            while abs(incorrect_azimuth - correct_azimuth) < 45:  # Ensure significant difference
                incorrect_azimuth = random.randint(-180, 180)
            incorrect_azimuths.append(incorrect_azimuth)
        
        incorrect_views = []
        for azimuth in incorrect_azimuths:
            view = self.create_perspective_view(video_frame, azimuth)
            incorrect_views.append(view)
        
        # Shuffle options
        all_views = [correct_view] + incorrect_views
        all_azimuths = [correct_azimuth] + incorrect_azimuths
        
        # Create random order
        order = list(range(4))
        random.shuffle(order)
        
        shuffled_views = [all_views[i] for i in order]
        shuffled_azimuths = [all_azimuths[i] for i in order]
        correct_answer = order.index(0)  # Find where the correct view ended up
        
        return {
            'question': f"Which image shows the correct view direction for the {event['class_name'].lower()} sound?",
            'audio_segment': audio_segment,
            'sample_rate': sr,
            'views': shuffled_views,
            'view_azimuths': shuffled_azimuths,
            'correct_answer': correct_answer,
            'event_info': {
                'class': event['class_name'],
                'timestamp': timestamp,
                'azimuth': event['azimuth'],
                'elevation': event['elevation'],
                'distance': event['distance']
            }
        }
    
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
            ok = cv2.imwrite(str(img_file), img[:, :, ::-1])
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
                'distance': int(question_data['event_info']['distance'])
            }
        }
        
        with open(output_path / f"{question_id}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def test_orientations(self, video_path: str, timestamp: float = 10.0, 
                         azimuth: float = 0.0) -> None:
        """Test different orientations to check which one looks correct"""
        frame = self.get_video_frame_at_time(video_path, timestamp)
        
        orientations = [
            ("Original", lambda x: x),
            ("Flip Vertical", lambda x: cv2.flip(x, 0)),
            ("Flip Horizontal", lambda x: cv2.flip(x, 1)),
            ("Flip Both", lambda x: cv2.flip(cv2.flip(x, 0), 1)),
            ("Rotate 180", lambda x: cv2.rotate(x, cv2.ROTATE_180))
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (name, transform) in enumerate(orientations):
            if i < len(axes):
                # Create perspective view with current orientation
                original_create = self.create_perspective_view
                
                def create_with_transform(frame, yaw, pitch=0, fov=90, output_size=(800, 600)):
                    # Temporarily modify the method
                    height, width = frame.shape[:2]
                    output_width, output_height = output_size
                    
                    x_map = np.zeros((output_height, output_width), dtype=np.float32)
                    y_map = np.zeros((output_height, output_width), dtype=np.float32)
                    
                    yaw_rad = math.radians(yaw)
                    pitch_rad = math.radians(pitch)
                    fov_rad = math.radians(fov)
                    
                    for y in range(output_height):
                        for x in range(output_width):
                            norm_x = (2.0 * x / output_width) - 1.0
                            norm_y = (2.0 * y / output_height) - 1.0
                            
                            z = 1.0
                            x_3d = norm_x * math.tan(fov_rad / 2)
                            y_3d = norm_y * math.tan(fov_rad / 2)
                            
                            length = math.sqrt(x_3d*x_3d + y_3d*y_3d + z*z)
                            x_3d /= length
                            y_3d /= length
                            z /= length
                            
                            x_rot = x_3d * math.cos(yaw_rad) + z * math.sin(yaw_rad)
                            z_rot = -x_3d * math.sin(yaw_rad) + z * math.cos(yaw_rad)
                            
                            y_rot = y_3d * math.cos(pitch_rad) - z_rot * math.sin(pitch_rad)
                            z_final = y_3d * math.sin(pitch_rad) + z_rot * math.cos(pitch_rad)
                            
                            longitude = math.atan2(x_rot, z_final)
                            latitude = math.asin(max(-1, min(1, y_rot)))
                            
                            u = (longitude + math.pi) / (2 * math.pi) * width
                            v = (math.pi/2 - latitude) / math.pi * height
                            
                            x_map[y, x] = u
                            y_map[y, x] = v
                    
                    perspective_frame = cv2.remap(frame, x_map, y_map, cv2.INTER_LINEAR)
                    return transform(perspective_frame)
                
                view = create_with_transform(frame, azimuth)
                view_rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
                
                axes[i].imshow(view_rgb)
                axes[i].set_title(f"{name}")
                axes[i].axis('off')
        
        # Hide empty subplot
        if len(orientations) < len(axes):
            axes[-1].axis('off')
        
        plt.suptitle(f"Orientation Test - Azimuth: {azimuth}°")
        plt.tight_layout()
        plt.savefig('orientation_test.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Orientation test saved as 'orientation_test.png'")
        print("Check which orientation looks most natural and update the create_perspective_view method accordingly.")
    
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

# Example usage
def main():
    # Initialize the question generator
    generator = STARSS23QuestionGenerator()
    
    # File paths (update these to your actual paths)
    audio_foa_path = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/STARSS23/foa_dev/dev-test-sony/fold4_room23_mix001.wav"
    audio_mic_path = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/STARSS23/mic_dev/dev-test-sony/fold4_room23_mix001.wav"
    video_path = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/STARSS23/video_dev/dev-test-sony/fold4_room23_mix001.mp4"
    metadata_path = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/STARSS23/metadata_dev/dev-test-sony/fold4_room23_mix001.csv"
    
    try:
        # Generate a question
        question_data = generator.generate_question_data(
            audio_foa_path, audio_mic_path, video_path, metadata_path
        )
        
        # Save the question (includes visualization)
        generator.save_question(question_data, "output/questions", "question_001")
        
        print(f"Question generated successfully!")
        print(f"Event: {question_data['event_info']}")
        print(f"Correct answer: Option {question_data['correct_answer'] + 1}")
        print("Files saved:")
        print("- question_001_audio.wav")
        print("- question_001_option_0.jpg to question_001_option_3.jpg") 
        print("- question_001_visualization.png")
        print("- question_001_metadata.json")
        
    except Exception as e:
        print(f"Error generating question: {e}")

if __name__ == "__main__":
    main()