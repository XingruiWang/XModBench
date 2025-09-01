import os
import random
import shutil
import wave
import contextlib
from pydub import AudioSegment
from PIL import Image
from tqdm import tqdm
import json


input_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/vggss_audio_bench"
output_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/temporal_order"
os.makedirs(output_dir, exist_ok=True)

# list all .wav files and map them to frame folders
wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
frame_dirs = [f for f in os.listdir(input_dir) if f.endswith("_frames") and f.split('_frames')[0]+'.wav' in wav_files]
frame_map = {f.split("_frames")[0]: f for f in frame_dirs}

# filter
wav_files = [f for f in wav_files if f.split('.wav')[0] in frame_map]
config_file_path = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/vggss.json"

with open(config_file_path, "r") as f:
    config = json.load(f)

id2class = {}
for item in config:
    id2class[item['file']] = item['class']
    
    

def get_event_id(wav_name):
    # Extract ID part before extension
    return wav_name.split(".")[0]

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
        if concat_images:
            widths, heights = zip(*(img.size for img in concat_images))
            total_width = sum(widths)
            max_height = max(heights)

            concat_img = Image.new("RGB", (total_width, max_height))
            x_offset = 0
            for img in concat_images:
                concat_img.paste(img, (x_offset, 0))
                x_offset += img.size[0]

            concat_img.save(os.path.join(choice_dir, "concat.png"))
        
        # Save order for this choice
        with open(os.path.join(choice_dir, "order.txt"), 'w') as f:
            f.write(','.join(orders))
            
# generate multiple samples
for i in tqdm(range(500)):
    sample_and_generate(i, num_instances=random.randint(3, 5))
