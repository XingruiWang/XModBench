import os
import random
import shutil
import wave
import contextlib
from pydub import AudioSegment
from PIL import Image
from tqdm import tqdm
import json
import glob
import re
import cv2
import subprocess
from pathlib import Path
import numpy as np
import multiprocessing as mp
from functools import partial

# Global configuration
input_dir = "/dockerx/local/data/audiobench/vggss/processed"
output_dir = "/dockerx/local/data/audiobench/temporal_order"
low_quality_file = '/dockerx/groups/AudioBench/low_av.txt'
config_file_path = "/dockerx/local/data/audiobench/vggss/vggss.json"

def load_global_data():
    """Load all the global data needed for processing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load low quality IDs
    low_quality_ids = []
    with open(low_quality_file, 'r') as f:
        for line in f:
            low_quality_ids.append(line.rstrip())
    
    # List all .wav files and map them to frame folders
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav") and f not in low_quality_ids]
    frame_dirs = [f for f in os.listdir(input_dir) if f.endswith("_frames") and f.split('_frames')[0]+'.wav' in wav_files and f not in low_quality_ids and len(os.listdir(os.path.join(input_dir, f))) >4]
    frame_map = {f.split("_frames")[0]: f for f in frame_dirs}
    
    # Filter wav files
    wav_files = [f for f in wav_files if f.split('.wav')[0] in frame_map]
    
    # Load config
    with open(config_file_path, "r") as f:
        config = json.load(f)
    
    id2class = {}
    for item in config:
        id2class[item['file']] = item['class']
    
    return wav_files, frame_map, id2class

def get_event_id(wav_name):
    """Extract ID part before extension"""
    return wav_name.split(".")[0]

def frames_to_mp4(frames, output_path, fps=24, duration=None, quiet=True):
    """
    Convert a sequence of frame images to MP4 video with H.264 encoding
    
    Args:
        frames: List of PIL Images
        output_path: Path for output MP4 file
        fps: Frames per second for output video (default: 25)
        quiet: Whether to suppress output messages
    """
    # Read first frame to get dimensions
    frames = [np.array(frame.convert("RGB")) for frame in frames]
    
    width = max(frames, key=lambda x: x.shape[1]).shape[1]
    height = max(frames, key=lambda x: x.shape[0]).shape[0]

    # Create temporary video file path
    output_path = Path(output_path)
    temp_path = output_path.parent / (output_path.stem + '_tmp.mp4')
    
    # Define the codec and create VideoWriter object for temporary video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
    

    for i in range(len(frames)):
        frame = frames[i]
        background = np.zeros((height, width, 3), dtype=np.uint8)
        x_offset = (width - frame.shape[1]) // 2
        y_offset = (height - frame.shape[0]) // 2
        background[y_offset:y_offset+frame.shape[0], x_offset:x_offset+frame.shape[1], :] = frame[:, :, ::-1]
        video_writer.write(background)
    
    # Release the video writer
    video_writer.release()
    cv2.destroyAllWindows()
    
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(temp_path),
        "-c:v", "libx264", "-crf", "23", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-loglevel", "error",
        str(output_path)
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()
        
        return True
        
    except subprocess.CalledProcessError as e:
        if not quiet:
            print(f"Error during H.264 encoding: {e}")
        return False
    except Exception as e:
        if not quiet:
            print(f"Unexpected error: {e}")
        return False

def sample_and_generate(idx, wav_files, frame_map, id2class, num_instances=3, num_choices=4):
    """Generate a single sample with all its choices"""
    try:
        # Set random seed for reproducibility within each process
        random.seed(idx + 12345)
        
        # Sample once from wav_files - these will be used for all choices
        samples = random.sample(wav_files, num_instances)
        
        sample_dir = os.path.join(output_dir, f"sample_{idx:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Prepare the base data (images, audio segments, orders) from the samples
        base_audio_segments = []
        base_images = []
        base_orders = []
        
        for wav_name in samples:
            wav_path = os.path.join(input_dir, wav_name)
            event_id = get_event_id(wav_name)

            # Load audio
            audio = AudioSegment.from_wav(wav_path)
            base_audio_segments.append(audio)

            # Load image
            video_id = event_id
            frame_folder = frame_map.get(video_id)

            if frame_folder:
                frames = []
                for frame_image in sorted(os.listdir(os.path.join(input_dir, frame_folder))):
                    frame_path = os.path.join(input_dir, frame_folder, frame_image)
                    frames.append(Image.open(frame_path))
                if len(frames) == 0:
                    print(f"Warning: No frame folder for {video_id}")
                    return False
                base_images.append(frames)
                
                base_orders.append(id2class[event_id].split(',')[0])
            else:
                print(f"Warning: No frame folder for {video_id}")
                return False
        
        # Create 4 different choices with different orders of the same samples
        already_have_choices = []
        for choice_num in range(num_choices):
            choice_dir = os.path.join(sample_dir, f"choice_{choice_num + 1}")
            os.makedirs(choice_dir, exist_ok=True)
            
            # Create a shuffled version of the indices
            indices = list(range(len(samples)))
            random.shuffle(indices)
            while tuple(indices) in already_have_choices:
                random.shuffle(indices)
            already_have_choices.append(tuple(indices))
            
            # Reorder audio, images, and orders based on shuffled indices
            audio_combined = AudioSegment.silent(duration=0)
            concat_frames = []
            orders = []
            
            for i, idx in enumerate(indices):
                # Add audio in this order
                audio_combined += base_audio_segments[idx]
                
                # Save individual event image
                frames = base_images[idx]

                interp_frames = []
                for j in range(48):
                    if j/48*len(frames) >= len(frames):
                        print(f"Warning: {j/48*len(frames)} is greater than {len(frames)}")
                    interp_frames.append(frames[int(j/48*len(frames))])
                concat_frames.extend(interp_frames)
                
                # Add order
                orders.append(base_orders[idx])

            # Export mixed wav for this choice
            mixed_path = os.path.join(choice_dir, "mixed.wav")
            audio_combined.export(mixed_path, format="wav")

            # Create concatenated image for this choice
            if concat_frames:
                widths, heights = zip(*(img.size for img in concat_frames))
                max_height = max(max(heights), 480)
                
                total_width = 0
                for i, (w, h) in enumerate(zip(widths, heights)):
                    new_height = max_height
                    new_width = int(new_height * w / h)
                    total_width += new_width
                    concat_frames[i] = concat_frames[i].resize((new_width, new_height))
            
            # Save to mp4 with 24 fps
            frames_to_mp4(concat_frames, os.path.join(choice_dir, "concat.mp4"))
            
            # Save order for this choice
            with open(os.path.join(choice_dir, "order.txt"), 'w') as f:
                f.write(','.join(orders))
        
        return True
        
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        return False

def process_batch(sample_indices, wav_files, frame_map, id2class):
    """Process a batch of samples in a single process"""
    successful = 0
    failed = 0
    
    for idx in tqdm(sample_indices, desc=f"Process {os.getpid()}"):
        num_instances = random.randint(3, 5)
        success = sample_and_generate(idx, wav_files, frame_map, id2class, num_instances)
        if success:
            successful += 1
        else:
            failed += 1
    
    return successful, failed

def parallel_processing(total_samples=500, num_processes=None):
    """
    Main function to run parallel processing
    
    Args:
        total_samples: Total number of samples to generate (default: 500)
        num_processes: Number of processes to use (default: CPU count)
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Loading global data...")
    wav_files, frame_map, id2class = load_global_data()
    print(f"Found {len(wav_files)} wav files")
    
    # Split sample indices across processes
    sample_indices = list(range(total_samples))
    batch_size = len(sample_indices) // num_processes
    batches = []
    
    for i in range(num_processes):
        start_idx = i * batch_size
        if i == num_processes - 1:  # Last process gets remaining samples
            end_idx = len(sample_indices)
        else:
            end_idx = (i + 1) * batch_size
        batches.append(sample_indices[start_idx:end_idx])
    
    print(f"Splitting {total_samples} samples across {num_processes} processes")
    for i, batch in enumerate(batches):
        print(f"Process {i+1}: samples {batch[0]} to {batch[-1]} ({len(batch)} samples)")
    
    # Create partial function with shared data
    process_func = partial(process_batch, 
                          wav_files=wav_files, 
                          frame_map=frame_map, 
                          id2class=id2class)
    
    # Run parallel processing
    print(f"Starting parallel processing with {num_processes} processes...")
    with mp.Pool(num_processes) as pool:
        results = pool.map(process_func, batches)
    
    # Aggregate results
    total_successful = sum(result[0] for result in results)
    total_failed = sum(result[1] for result in results)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {total_successful}/{total_samples}")
    print(f"Failed: {total_failed}/{total_samples}")

if __name__ == "__main__":
    # You can customize these parameters
    TOTAL_SAMPLES = 500
    NUM_PROCESSES = 32 # Set to None to use all CPU cores
    
    parallel_processing(total_samples=TOTAL_SAMPLES, num_processes=NUM_PROCESSES)