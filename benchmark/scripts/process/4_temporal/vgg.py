import os
import cv2
import json
import subprocess
import shutil
import librosa
import librosa.display
import matplotlib.pyplot as plt

root = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/ExtremeCountixAV"
extreme_label = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/ExtremeCountixAV/ExtremeLabels.csv"
output_root ="/home/xwang378/scratch/2025/AudioBench/benchmark/Data/ExtremCountAV"


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

video_paths = {}
audio_paths = {}
for class_name in os.listdir(os.path.join(root, "Videos")):
    for video_id in os.listdir(os.path.join(root, "Videos", class_name)):
        video_paths[video_id.split(".mp4")[0]] = os.path.join(root, "Videos", class_name, f"{video_id.split('.mp4')[0]}.mp4")
        audio_paths[video_id.split(".mp4")[0]] = os.path.join(root, "Audio", class_name, f"{video_id.split('.mp4')[0]}.wav")

for video_id in videos:
    video_info = videos[video_id]
    video_path = video_paths[video_id]
    audio_path = audio_paths[video_id]
    
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
    # output_path = os.path.join(output_root, f"{video_id}.mp4")
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # frame_idx = 0
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     if frame_idx >= start_frame and frame_idx <= end_frame:
    #         out.write(frame)

    #     frame_idx += 1
    #     if frame_idx > end_frame:
    #         break

    # cap.release()
    # out.release()
    # print(f"Cropped video saved: {output_path}")
    
    with open(os.path.join(output_root, f"{video_id}.json"), "w") as f:
        json.dump(video_info, f)
        
    # copy_audio
    shutil.copy(audio_path, os.path.join(output_root, f"{video_id}.wav"))

    # copy_video
    shutil.copy(video_path, os.path.join(output_root, f"{video_id}.mp4"))
    

    # # save_audio
    # # Step 1: Extract audio from original video
    # if '-0HwkO7TRmc' in video_id:
    #     import ipdb; ipdb.set_trace()
    
    # load audio
    audio, sr = librosa.load(audio_path, sr=None)
    
    # load video
    
    # # Step 2: Trim audio to match the cropped frame segment
    # start_time_sec = int(start_frame) / fps
    # if end_frame == -1:
    #     end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # duration_sec = (int(end_frame) - int(start_frame)) / fps
    trimmed_audio_path = os.path.join(output_root, f"{video_id}_audio.wav")
    remove_audio_cmd = [
        "rm", trimmed_audio_path
    ]
    subprocess.run(remove_audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # print(f"start_time_sec: {start_time_sec}, duration_sec: {duration_sec}")
    # trim_audio_cmd = [
    #     "ffmpeg", "-y", "-i", audio_path,
    #     "-ss", str(start_time_sec),
    #     "-t", str(duration_sec),
    #     trimmed_audio_path
    # ]
    # subprocess.run(trim_audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # print(f"Trimmed audio saved: {trimmed_audio_path}")
    
    # plot spectrogram and save
    audio, sr = librosa.load(audio_path, sr=None)
    spectrogram = librosa.stft(audio)
    # 使用"viridis"色卡（黄绿色调）
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title(f"Spectrogram")
    plt.savefig(os.path.join(output_root, f"{video_id}.png"))
    plt.close()
    print(f"Spectrogram saved: {os.path.join(output_root, f'{video_id}.png')}")
    # import ipdb; ipdb.set_trace()
    