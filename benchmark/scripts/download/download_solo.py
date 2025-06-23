import os
import subprocess
import json
from multiprocessing import Pool, cpu_count

SAVE_DIR = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/solos/videos"
JSON_PATH = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/solos/solos.json"

with open(JSON_PATH, "r") as f:
    video_dict = json.load(f)

# Flatten the dictionary into a list of (category, video_id) pairs
tasks = []
for category, video_ids in video_dict.items():
    save_path = os.path.join(SAVE_DIR, category)
    os.makedirs(save_path, exist_ok=True)
    for vid in video_ids:
        output_path = os.path.join(save_path, f"{vid}.mp4")
        if not os.path.exists(output_path):  # Skip if file exists
            tasks.append((category, vid))

def download_video(task):
    category, vid = task
    save_path = os.path.join(SAVE_DIR, category)
    output_path = os.path.join(save_path, f"{vid}.mp4")
    url = f"https://www.youtube.com/watch?v={vid}"
    result = subprocess.run([
        "yt-dlp",
        "-f", "mp4",
        "-o", output_path,
        url
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if result.returncode == 0:
        print(f"[✓] {vid}")
    else:
        print(f"[✗] {vid}")

if __name__ == "__main__":
    with Pool(processes=1) as pool:
        pool.map(download_video, tasks)
