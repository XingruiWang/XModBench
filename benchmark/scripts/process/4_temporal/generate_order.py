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

input_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/vggss_audio_bench"
output_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/temporal_order"
os.makedirs(output_dir, exist_ok=True)

low_quality_file = '/home/xwang378/scratch/2025/AudioBench/benchmark/Data/vggss_meta_json/low_av.txt'

low_quality_ids = []
with open(low_quality_file, 'r') as f:
    for line in f:
        low_quality_ids.append(line.rstrip())

# list all .wav files and map them to frame folders
wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav") and f not in low_quality_ids]
frame_dirs = [f for f in os.listdir(input_dir) if f.endswith("_frames") and f.split('_frames')[0]+'.wav' in wav_files and f not in low_quality_ids]
frame_map = {f.split("_frames")[0]: f for f in frame_dirs}

# filter
wav_files = [f for f in wav_files if f.split('.wav')[0] in frame_map]

config_file_path = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/vggss_meta_json/vggss.json"

with open(config_file_path, "r") as f:
    config = json.load(f)

id2class = {}
for item in config:
    id2class[item['file']] = item['class']
    
def get_event_id(wav_name):
    # Extract ID part before extension
    return wav_name.split(".")[0]


def frames_to_mp4(frames, output_path, fps=1.0, quiet=True):
    """
    Convert a sequence of frame images to MP4 video with H.264 encoding
    
    Args:
        frames_folder: Path to folder containing frame images
        output_path: Path for output MP4 file
        fps: Frames per second for output video (default: 2.0)
        quiet: Whether to suppress output messages
    """
   
    
    # Read first frame to get dimensions
    frames = [np.array(frame.convert("RGB")) for frame in frames]
    
    width= max(frames, key=lambda x: x.shape[1]).shape[1]
    height = max(frames, key=lambda x: x.shape[0]).shape[0]


    # Create temporary video file path
    output_path = Path(output_path)
    temp_path = output_path.parent / (output_path.stem + '_tmp.mp4')
    
    # Define the codec and create VideoWriter object for temporary video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
    
    # Process each frame
    for i, frame in enumerate(frames):   
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
        "-pix_fmt", "yuv420p",  # Ensure compatibility
        "-loglevel", "error",   # Only show errors
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
    
def sample_and_generate(idx, num_instances=3, num_choices=4):
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
            first_image = os.listdir(os.path.join(input_dir, frame_folder))[0]
            frame_path = os.path.join(input_dir, frame_folder, first_image)
            img = Image.open(frame_path)
            base_images.append(img)
            base_orders.append(id2class[event_id].split(',')[0])
        else:
            print(f"Warning: No frame folder for {video_id}")
    
    # Create 4 different choices with different orders of the same samples
    for choice_num in range(num_choices):
        choice_dir = os.path.join(sample_dir, f"choice_{choice_num + 1}")
        os.makedirs(choice_dir, exist_ok=True)
        
        # Create a shuffled version of the indices
        indices = list(range(len(samples)))
        random.shuffle(indices)
        while indices == list(range(len(samples))):
            print("Shuffling again")
            random.shuffle(indices)
        
        # Reorder audio, images, and orders based on shuffled indices
        audio_combined = AudioSegment.silent(duration=0)
        concat_images = []
        orders = []
        
        for i, idx in enumerate(indices):
            # Add audio in this order
            audio_combined += base_audio_segments[idx]
            
            # Save individual event image
            img = base_images[idx]
            target_img_path = os.path.join(choice_dir, f"event{i+1}.png")
            img.save(target_img_path)
            concat_images.append(img)
            
            # Add order
            orders.append(base_orders[idx])

        # Export mixed wav for this choice
        mixed_path = os.path.join(choice_dir, "mixed.wav")
        audio_combined.export(mixed_path, format="wav")

        # Create concatenated image for this choice
        print(f"Creating concatenated image for {choice_dir}, {len(concat_images)} frames")
        if concat_images:
            widths, heights = zip(*(img.size for img in concat_images))

            max_height = max(max(heights), 480)
            
            total_width = 0
            for i, (w, h) in enumerate(zip(widths, heights)):
                new_height = max_height
                
                new_width = int(new_height * w / h)
                total_width += new_width
                concat_images[i] = concat_images[i].resize((new_width, new_height))
            
            bin_width = 10
            total_width += bin_width * (len(concat_images) - 1)
            concat_img = Image.new("RGB", (total_width, max_height))
            
            x_offset = 0
            for img in concat_images:
                concat_img.paste(img, (x_offset, 0))
                x_offset += img.size[0]
                x_offset += bin_width

            concat_img.save(os.path.join(choice_dir, "concat.png"))
        
        # save to mp4 with 2 fps
        frames_to_mp4(concat_images, os.path.join(choice_dir, "concat.mp4"))
        
        # Save order for this choice
        with open(os.path.join(choice_dir, "order.txt"), 'w') as f:
            f.write(','.join(orders))
            
# generate multiple samples
for i in tqdm(range(500)):
    sample_and_generate(i, num_instances=random.randint(3, 5))
