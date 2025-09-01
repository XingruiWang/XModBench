import os
import cv2
import librosa
import numpy as np
import soundfile as sf
from moviepy.editor import VideoFileClip
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# root_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/solos/videos"
# output_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/solos_processed"

root_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/urmp"
output_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/urmp_processed"


# root_dir = "/dockerx/local/data/Landscapes/landscape/train"
# output_dir = "/dockerx/local/data/Landscapes/landscape/train_processed"

os.makedirs(output_dir, exist_ok=True)

# videos_dir = os.path.join(root_dir, "videos")
videos_dir = root_dir

def extract_middle_frame(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
    cap.release()

def extract_peak_audio_segment(video_path, output_path, sr=16000, window=2.0):


    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path
    ])
    
    audio_data, sr = sf.read(output_path)
    mono_audio = audio_data if audio_data.ndim == 1 else audio_data.mean(axis=1)

    win_len = int(sr * window)
    max_energy = -np.inf
    best_start = 0
    for start in range(0, len(mono_audio) - win_len, len(mono_audio) // 32):
        segment = mono_audio[start:start + win_len]
        energy = np.mean(segment ** 2)
        if energy > max_energy:
            max_energy = energy
            best_start = start
    peak_audio = mono_audio[best_start:best_start + win_len]
    sf.write(output_path, peak_audio, sr)

def process_one(args):
    video_path, img_out, wav_out = args
    try:
        extract_middle_frame(video_path, img_out)
        extract_peak_audio_segment(video_path, wav_out)
    except Exception as e:
        print(f"[Error] {video_path}: {e}")

def collect_jobs():
    jobs = []
    for category in os.listdir(videos_dir):
        category_path = os.path.join(videos_dir, category)
        if not os.path.isdir(category_path):
            continue
        out_cat_dir = os.path.join(output_dir, category)
        os.makedirs(out_cat_dir, exist_ok=True)

        for vid_file in os.listdir(category_path):
            if not vid_file.endswith(".mp4"):
                continue
            video_path = os.path.join(category_path, vid_file)
            video_id = os.path.splitext(vid_file)[0]
            img_out = os.path.join(out_cat_dir, f"{video_id}.jpg")
            wav_out = os.path.join(out_cat_dir, f"{video_id}.wav")
            jobs.append((video_path, img_out, wav_out))
    return jobs

if __name__ == "__main__":

    jobs = collect_jobs()
    with Pool(processes=min(cpu_count(), 8)) as pool: 
        list(tqdm(pool.imap_unordered(process_one, jobs), total=len(jobs)))
