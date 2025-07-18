import os
import cv2
import json
import subprocess

Vgg_root = "/home/vdd/scratch/2025/AudioBench/benchmark/Data/ExtremeCountixAV/VGG_Sound_Dataset"
extreme_label = "/home/vdd/scratch/2025/AudioBench/benchmark/Data/ExtremeCountixAV/extreme_label.csv"
output_root = "/home/vdd/scratch/2025/AudioBench/benchmark/Data/ExtremeCountixAV/Cropped_Videos"
os.makedirs(output_root, exist_ok=True)

# 读取标签文件
videos = {}
with open(extreme_label, "r") as f:
    for line in f:
        youtube_id, repetition_start_frame, repetition_end_frame, start_crop_frame, end_crop_frame, number_of_repetitions, action_class = line.strip().split()
        videos[youtube_id] = {
            "youtube_id": youtube_id,
            "repetition_start_frame": int(repetition_start_frame),
            "repetition_end_frame": int(repetition_end_frame),
            "start_crop_frame": int(start_crop_frame),
            "end_crop_frame": int(end_crop_frame),
            "number_of_repetitions": int(number_of_repetitions),
            "action_class": action_class
        }

# 遍历视频
for video_id in videos:
    video_info = videos[video_id]
    pattern = os.path.join(Vgg_root, f"{video_id}.*")
    matched_files = [f for f in os.listdir(Vgg_root) if f.startswith(video_id)]
    
    if not matched_files:
        print(f"Video {video_id} not found!")
        continue

    video_path = os.path.join(Vgg_root, matched_files[0])
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = video_info["start_crop_frame"]
    end_frame = video_info["end_crop_frame"]

    # 输出路径
    output_path = os.path.join(output_root, f"{video_id}_cropped.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= start_frame and frame_idx <= end_frame:
            out.write(frame)

        frame_idx += 1
        if frame_idx > end_frame:
            break

    cap.release()
    out.release()
    print(f"Cropped video saved: {output_path}")
    
    with open(os.path.join(output_root, f"{video_id}.json"), "w") as f:
        json.dump(video_info, f)
        
    # save_audio
    subprocess.run([
        "ffmpeg", "-y", "-i", output_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path.replace(".mp4", ".wav")
    ])
