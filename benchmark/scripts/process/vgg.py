import os
import cv2
import json
import subprocess

Vgg_root = "/dockerx/local/data/VGGSound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video"
extreme_label = "/dockerx/share/AudioBench/benchmark/scripts/process/ExtremeLabels.csv"
output_root = "/dockerx/share/AudioBench/Cropped_Videos"
os.makedirs(output_root, exist_ok=True)

# 读取标签文件
videos = {}
with open(extreme_label, "r") as f:
    for line in f:
        if line.startswith("youtube_id"):
            continue
        try:
            youtube_id, repetition_start_frame, repetition_end_frame, start_crop_frame, end_crop_frame, number_of_repetitions, action_class = line.strip().split(',')
        except ValueError:
            youtube_id, repetition_start_frame, repetition_end_frame, start_crop_frame, end_crop_frame, number_of_repetitions = line.strip().split(',')
            action_class = "unknown"

        videos[youtube_id] = {
            "youtube_id": youtube_id,
            "repetition_start_frame": int(repetition_start_frame),
            "repetition_end_frame": int(repetition_end_frame),
            "start_crop_frame": int(start_crop_frame),
            "end_crop_frame": int(end_crop_frame),
            "number_of_repetitions": int(float(number_of_repetitions)),
            "action_class": action_class
        }

# 遍历视频
for video_id in videos:
    video_info = videos[video_id]
    pattern = os.path.join(Vgg_root, f"{video_id}.*")
    matched_files = [f for f in os.listdir(Vgg_root) if f.startswith(video_id)]
    
    if not matched_files:
        # print(f"Video {video_id} not found!")
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
    
    # with open(os.path.join(output_root, f"{video_id}.json"), "w") as f:
    #     json.dump(video_info, f)
        
    # # save_audio
    # # Step 1: Extract audio from original video
    # audio_path = os.path.join(output_root, f"{video_id}_audio.wav")
    # extract_audio_cmd = [
    #     "ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", audio_path
    # ]
    # subprocess.run(extract_audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # # Step 2: Trim audio to match the cropped frame segment
    # start_time_sec = int(start_frame) / fps
    # duration_sec = (int(end_frame) - int(start_frame)) / fps
    # trimmed_audio_path = os.path.join(output_root, f"{video_id}_audio.wav")

    # trim_audio_cmd = [
    #     "ffmpeg", "-y", "-i", audio_path,
    #     "-ss", str(start_time_sec),
    #     "-t", str(duration_sec),
    #     trimmed_audio_path
    # ]
    # subprocess.run(trim_audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)