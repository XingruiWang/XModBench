import cv2
import os
import pandas as pd
import numpy as np


video_root = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/Urbansas/video_2fps"
audio_root = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/Urbansas/audio"
output_root = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/urbansas_samples_videos"
os.makedirs(output_root, exist_ok=True)

csv_path = "/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/process/2_spatial_audio/audio_with_motion_and_visibility.csv"
spatial_audio_df = pd.read_csv(csv_path)

def uniform_sample(images_list, k=6):
    n = len(images_list)
    if n <= 8:
        return images_list  
    indices = np.linspace(0, n - 1, k).astype(int)
    return [images_list[i] for i in indices]

def extract_clip_audio_video(clip_id, start_time, end_time, fps=2):
    video_path = os.path.join(video_root, f"{clip_id}.mp4")
    audio_path = os.path.join(audio_root, f"{clip_id}.wav")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None, None, None

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # video_output_path = os.path.join(output_root, f"{clip_id}_{start_time:.2f}_{end_time:.2f}.mp4")
    # out_video = cv2.VideoWriter(video_output_path, fourcc, 2.0, (width, height))

    frame_list = []
    images_list = []
    for fid in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if ret:
            # out_video.write(np.zeros((height, width, 3), dtype=np.uint8))
            frame_list.append(fid)
            images_list.append(frame)
            os.makedirs(os.path.join(output_root, f"{clip_id}_{start_time:.2f}_{end_time:.2f}"), exist_ok=True)
            cv2.imwrite(os.path.join(output_root, f"{clip_id}_{start_time:.2f}_{end_time:.2f}/frame_{fid}.jpg"), frame)
    cap.release()
    # out_video.release()
    
    # concat
    
    
    
    images_list = uniform_sample(images_list)
    images_concat = np.concatenate(images_list, axis=1)
    # resize
    height, width  = images_list[0].shape[:2]
    images_concat = cv2.resize(images_concat,(width * 3 ,  height * 3 // len(frame_list)))
    cv2.imwrite(os.path.join(output_root, f"{clip_id}_{start_time:.2f}_{end_time:.2f}_frames.jpg"), images_concat)
    
    # save video
    video_output_path = os.path.join(output_root, f"{clip_id}_{start_time:.2f}_{end_time:.2f}.mp4")
    out_video = cv2.VideoWriter(video_output_path, fourcc, 2.0, (width, height))
    for frame in images_list:
        out_video.write(frame)
    out_video.release()

    # Audio
    audio_output_path = os.path.join(output_root, f"{clip_id}_{start_time:.2f}_{end_time:.2f}.wav")
    ffmpeg_cmd = (
        f"ffmpeg -y -i '{audio_path}' -ss {start_time} -to {end_time} -c copy '{audio_output_path}'"
    )
    os.system(ffmpeg_cmd)
    
    

    return audio_output_path, output_root, frame_list

results = []
for i, row in spatial_audio_df.iterrows():
    clip_id = row["filename"]
    label = row["label"]
    start = float(row["start"])
    end = float(row["end"])
    class_id = row["class_id"]
    label = row["label"]
    direction_x = float(row["direction_x"])
    direction_y = float(row["direction_y"])
    # import ipdb; ipdb.set_trace()
    
    audio_path, video_path, frames = extract_clip_audio_video(clip_id, start, end)
    if audio_path and video_path:
        results.append({
            "clip_id": clip_id,
            "class_id": class_id,
            "label": label,
            "start": start,
            "end": end,
            "direction_x": direction_x,
            "direction_y": direction_y,
            "class_id": class_id,
            "audio_clip": audio_path,
            "video_clip": video_path,
            "frames": frames
        })

output_df = pd.DataFrame(results)
output_csv = os.path.join("urbansas_extracted_samples.csv")
output_df.to_csv(output_csv, index=False)
