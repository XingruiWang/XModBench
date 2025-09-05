#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
spatial_audio_processor.py - Process spatial audio files with augmentation and filter by azimuth

This script processes audio files in the STARSS23 dataset structure:
1. Applies spatial augmentation to WAV files using the quad_binaural renderer
2. Calculates final azimuth values based on event info and choice rotations
3. Filters events within specific azimuth ranges: (-20, 20), (70, 110), (160, 200), (250, 290)
4. Outputs qualifying event names to high_quality.txt
5. Adds bounding boxes to video files before saving
6. Uses multiprocessing for parallel event folder processing while keeping subfolder processing sequential
"""

import os
import json
import glob
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, butter, filtfilt
import pysofaconventions
from rotate_audio import rotate_audio_azimuth_mic, rotate_video_azimuth, save_video_clip
import librosa
import cv2
import tempfile
import subprocess
from typing import Tuple
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


mic_dirs = np.array([
                        [ 0.78539816,  0.61086524],
                        [-0.78539816, -0.61086524],
                        [ 2.35619449, -0.61086524],
                        [-2.35619449,  0.61086524]
                    ])

# ==================== BBox Rendering Helpers ====================

def azimuth_to_video_position(azimuth: float, elevation: float,
                              video_width: int, video_height: int) -> Tuple[int, int, int, int]:
    """
    Map (azimuth, elevation) to bbox in an equirectangular 360° frame.
    CCW positive azimuth, 0° = front; elevation in [-90, 90] (up positive).
    Returns (x1, y1, x2, y2).
    """
    W, H = video_width, video_height

    # 中心点 X: azimuth -> 水平方向
    center_x = int((0.5 - azimuth / 360.0) * W) % W

    # 中心点 Y: elevation -> 垂直方向 (+90=顶, 0=中, -90=底)
    el = max(-90.0, min(90.0, elevation))
    center_y = int((0.5 - el / 180.0) * H)

    # 框大小（可以调大一些保证覆盖）
    bbox_width = W // 6
    bbox_height = H // 3

    x1 = max(0, center_x - bbox_width // 2)
    x2 = min(W - 1, center_x + bbox_width // 2)
    y1 = max(0, center_y - bbox_height // 2)
    y2 = min(H - 1, center_y + bbox_height // 2)

    return (x1, y1, x2, y2)

def draw_direction_bbox(frame: np.ndarray, azimuth: int, elevation: int, event_class: str) -> np.ndarray:
    """Draw a simple red bounding box on the video frame"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = azimuth_to_video_position(azimuth, elevation, w, h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red box
    cv2.putText(frame, event_class, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame

def add_bbox_to_video(input_video_path: str, output_video_path: str, azimuth: int, elevation: int, event_class: str):
    """
    Add red bounding box to a video file.
    Internally uses OpenCV to write mp4v, then re-encodes with ffmpeg to H.264.
    """
    cap = cv2.VideoCapture(str(input_video_path))  # Convert Path to string
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 临时 mp4v - Fix the Path issue here
    output_path = Path(output_video_path)
    temp_path = output_path.parent / (output_path.stem + '_tmp.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_with_bbox = draw_direction_bbox(frame, azimuth, elevation, event_class)
        out.write(frame_with_bbox)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[bbox] Processing frame {frame_count}/{total_frames}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Temporary mp4v video saved: {temp_path}")

    # 统一转码为 H.264
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(temp_path),
        "-c:v", "libx264", "-crf", "23", "-preset", "medium",
        "-an",
        "-loglevel", "quiet",
        str(output_video_path)
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    
    print(f"[INFO] Final H.264 video saved to: {output_video_path}")
    
    temp_path.unlink()  # Remove temp file using Path method
    print(f"[INFO] Cleanup complete")

def save_video_clip_with_bbox(rotated_video_choice, output_file, azimuth, elevation=0, event_class=""):
    """
    Save video clip with bounding box overlay.
    This replaces the original save_video_clip call.
    """
    # First save the rotated video to a temporary file
    temp_output = output_file.parent / (output_file.stem + '_temp.mp4')
    
    # Save the rotated video first (assuming save_video_clip exists in rotate_audio module)
    save_video_clip(rotated_video_choice, temp_output)
    
    # Then add bounding box and save to final location
    add_bbox_to_video(temp_output, output_file, azimuth, elevation, event_class)
    
    # Clean up temporary file
    if temp_output.exists():
        temp_output.unlink()

# Import functions from the original quad_binaural module
def enhance_front_back(L, R, az_deg, sr):
    """增强前后位置的频谱特征差异"""
    # 后方声源添加轻微高频衰减（模拟头部遮挡）
    if abs(az_deg) > 90:  # 后方
        b, a = butter(2, 8000/(sr/2), 'low')
        L = filtfilt(b, a, L) * 0.6
        R = filtfilt(b, a, R) * 0.6
    
    # 前方声源增强中高频清晰度
    else:  # 前方
        b, a = butter(2, [2000/(sr/2), 6000/(sr/2)], 'band')
        enhancement = filtfilt(b, a, L) * 0.2
        L = L + enhancement
        enhancement = filtfilt(b, a, R) * 0.2
        R = R + enhancement
    
    return L, R

def wrap180(x_deg):
    """Wrap angle to [-180, 180] degrees"""
    return (np.asarray(x_deg) + 180.0) % 360.0 - 180.0

def as_degrees(az, el):
    az = az.astype(float) if isinstance(az, np.ndarray) else float(az)
    el = el.astype(float) if isinstance(el, np.ndarray) else float(el)
    # Heuristic: if mostly within +/-3.2, assume radians -> convert
    if isinstance(az, np.ndarray):
        if np.percentile(np.abs(az), 95) < 3.2 and np.percentile(np.abs(el), 95) < 3.2:
            az = np.degrees(az); el = np.degrees(el)
    else:
        if abs(az) < 3.2 and abs(el) < 3.2:
            az = np.degrees(az); el = np.degrees(el)
    return az, el

def get_IR_MRN(sofa):
    """Return HRIR array of shape [M, 2, N]."""
    IR = sofa.getDataIR()
    if IR is not None and np.size(IR) > 0:
        return IR
    real = sofa.getDataReal()
    imag = sofa.getDataImag()
    if real is None or imag is None:
        raise ValueError("SOFA lacks Data.IR and (Data.Real, Data.Imag).")
    H = real + 1j * imag  # [M, 2, F]
    IR = np.fft.irfft(H, axis=-1)  # -> [M, 2, N]
    return IR

def nearest_idx_for(sofa, az_deg_tgt, el_deg_tgt):
    """Find nearest SOFA measurement index by (az, el) in degrees."""
    pos = sofa.getVariableValue('SourcePosition')  # [M, 3] (az, el, r)
    az = pos[:, 0].astype(float)
    el = pos[:, 1].astype(float)
    az, el = as_degrees(az, el)
    # Use wrap for azimuth difference
    d_az = wrap180(az - az_deg_tgt)
    d_el = (el - el_deg_tgt)
    idx = int(np.argmin(d_az**2 + d_el**2))
    return idx

def get_hrir(sofa, az_deg, el_deg=0.0):
    IR = get_IR_MRN(sofa)  # [M, 2, N], receivers: 0=Left,1=Right
    idx = nearest_idx_for(sofa, az_deg, el_deg)
    hL = IR[idx, 0, :]
    hR = IR[idx, 1, :]
    if not np.any(np.abs(hL)) and not np.any(np.abs(hR)):
        raise ValueError(f"HRIR at idx={idx} (az={az_deg}, el={el_deg}) is all zeros.")
    return hL, hR

def convolve_to_stereo(x_mono, hL, hR):
    L = fftconvolve(x_mono, hL, mode="full")
    R = fftconvolve(x_mono, hR, mode="full")
    return L, R

def apply_spatial_augmentation(audio_data, azimuth, sofa_path, sr):
    """Apply spatial augmentation to audio data based on azimuth"""
    try:
        sofa = pysofaconventions.SOFAFile(sofa_path, "r")
        if audio_data.shape[1] > 100:
            audio_data = np.transpose(audio_data, (1, 0))
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_mono = np.mean(audio_data, axis=1)
        else:
            audio_mono = audio_data
            
        # Get HRIR for the specified azimuth
        hL, hR = get_hrir(sofa, azimuth, 0.0)
        
        # Convolve with HRIR
        L, R = convolve_to_stereo(audio_mono, hL, hR)
        
        # Apply front/back enhancement
        L, R = enhance_front_back(L, R, azimuth, sr)
        
        # Combine to stereo
        stereo_audio = np.stack([L, R], axis=1)
        
        # Normalize
        peak = max(np.max(np.abs(L)), np.max(np.abs(R)), 1e-12)
        target = 10.0 ** (-1.0 / 20.0)  # -1 dBFS
        gain = min(1.0, target / peak)
        stereo_audio = stereo_audio * gain
        
        sofa.close()
        return stereo_audio.astype(np.float32)
        
    except Exception as e:
        print(f"Warning: Spatial augmentation failed for azimuth {azimuth}: {e}")
        # Return original audio as stereo if augmentation fails
        if len(audio_data.shape) == 1:
            return np.stack([audio_data, audio_data], axis=1).astype(np.float32)
        return audio_data.astype(np.float32)

def is_in_quality_range(azimuth):
    """Check if azimuth is in one of the high quality ranges"""
    # Normalize azimuth to [0, 360) for easier range checking
    az = azimuth % 360
    
    quality_ranges = [
        (340, 360),  # (-20, 0) mapped to (340, 360)
        (0, 20),     # (0, 20)
        (70, 110),   # (70, 110)  
        (160, 200),  # (160, 200)
        (250, 290)   # (250, 290)
    ]
    
    for start, end in quality_ranges:
        if start <= az <= end:
            return True
    return False

def sample_high_quality_azimuth():
    import random
    quality_ranges = [
        (340, 360),  # (-20, 0) mapped to (340, 360)
        (0, 20),     # (0, 20)
        (70, 90),   # (70, 110)  
        (270, 290)   # (250, 290)
    ]

    random_range = random.sample(quality_ranges, 1)[0]
    random_azimuth = random.uniform(random_range[0], random_range[1])
    return int(random_azimuth)

def process_audio_folder(folder_path, sofa_path, output_dir):
    """Process all audio files in a folder"""
    try:
        folder_path = Path(folder_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        high_quality_events = []
        
        # Find all metadata files
        metadata_files = list(folder_path.glob("*_metadata.json"))
        
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            event_name = metadata_file.stem.replace("_metadata", "")
            print(f"[PID:{os.getpid()}] Processing event: {event_name}")
            
            output_metadata_file = output_dir / f"{event_name}_metadata.json"
            if output_metadata_file.exists():
                print(f"  Metadata file already exists: {output_metadata_file}")
                mp4_files = list(output_dir.glob(f"{event_name}*.mp4"))
                mp4_files = [mp4_file for mp4_file in mp4_files if 'temp' not in mp4_file.name or 'tmp' not in mp4_file.name]
                wav_files = list(output_dir.glob(f"{event_name}*.wav"))
                wav_files = [wav_file for wav_file in wav_files if 'temp' not in wav_file.name or 'tmp' not in wav_file.name]
                if len(wav_files) + len(mp4_files) >= 5:
                    print(f"  Event {event_name} has already 5 files, skipping")
                    continue
            
            # Get event info
            event_info = metadata.get("event_info", {})
            base_azimuth = event_info.get("azimuth", 0)
            elevation = event_info.get("elevation", 0)  # Get elevation for bbox
            event_class = event_info.get("class", "unknown")  # Get event class for bbox
            add_on_azimuth = 0  # Initialize add_on_azimuth
            
            # Process input audio if it's video_choice type
            if metadata.get("question_type") == "video_choice":
                input_audio_file = folder_path / f"{event_name}_input_audio.wav"
                if input_audio_file.exists():
                    audio_data, sr = sf.read(input_audio_file)
                    # Check if base azimuth is in quality range
                    if is_in_quality_range(base_azimuth):
                        high_quality_events.append(event_name)
                        print(f"  Event {event_name} added to high quality (azimuth: {base_azimuth})")
                    else:
                        high_quality_azimuth = sample_high_quality_azimuth()
                        add_on_azimuth = high_quality_azimuth - base_azimuth
                        base_azimuth = high_quality_azimuth
                        audio_data = rotate_audio_azimuth_mic(audio_data, mic_dirs, add_on_azimuth)
                        event_info["azimuth"] = base_azimuth
                        metadata_file_out = output_dir / f"{event_name}_metadata.json"
                        with open(metadata_file_out, 'w') as f:
                            json.dump(metadata, f, indent=4)
                        
                    augmented_audio = apply_spatial_augmentation(audio_data, base_azimuth, sofa_path, sr)
                    
                    output_file = output_dir / f"{event_name}_input_audio.wav"
                    sf.write(output_file, augmented_audio, sr)
                    print(f"  Processed input audio -> {output_file}")
                    
                # rotate video choice
                video_choices = [folder_path / f"{event_name}_choice_{i}.mp4" for i in range(4)]
                for i, video_choice in enumerate(video_choices):
                    if video_choice.exists():
                        rotated_video_choice = rotate_video_azimuth(video_choice, add_on_azimuth)
                        output_file = output_dir / f"{event_name}_choice_{i}.mp4"
                        # Use the new function that adds bbox before saving
                        save_video_clip_with_bbox(rotated_video_choice, output_file, 
                                                metadata.get("choice_video_rotations", [0])[i] + base_azimuth, elevation, event_class)
                        print(f"  Processed video choice with bbox -> {output_file}")
                    else:
                        print(f"  Video choice {video_choice} does not exist")
                        raise ValueError(f"Video choice {video_choice} does not exist")
                
                # Remove the ipdb.set_trace() line
                print(f"  Video processing complete for {event_name}")
            
            # Process choice audio files
            choice_rotations = metadata.get("choice_audio_rotations", [])
            if is_in_quality_range(base_azimuth):
                if event_name not in high_quality_events:
                    high_quality_events.append(event_name)
            else:
                high_quality_azimuth = sample_high_quality_azimuth()
                add_on_azimuth = high_quality_azimuth - base_azimuth
                base_azimuth = high_quality_azimuth
                event_info["azimuth"] = base_azimuth
                metadata_file_out = output_dir / f"{event_name}_metadata_add_on.json"
                with open(metadata_file_out, 'w') as f:
                    json.dump(metadata, f, indent=4)
                            
                for i, rotation in enumerate(choice_rotations):
                    choice_file = folder_path / f"{event_name}_choice_{i}.wav"
                    if choice_file.exists():
                        audio_data, sr = sf.read(choice_file)
                        audio_data = rotate_audio_azimuth_mic(audio_data, mic_dirs, add_on_azimuth)
                        
                        final_azimuth = base_azimuth + rotation
                        
                        augmented_audio = apply_spatial_augmentation(audio_data, final_azimuth, sofa_path, sr)
                        
                        output_file = output_dir / f"{event_name}_choice_{i}.wav"
                        sf.write(output_file, augmented_audio, sr)
                        print(f"  Processed choice {i} (azimuth: {final_azimuth}) -> {output_file}")
                        
                video_input_file = folder_path / f"{event_name}_input_video.mp4"
                if video_input_file.exists():
                    rotated_video_input = rotate_video_azimuth(video_input_file, add_on_azimuth)
                    output_file = output_dir / f"{event_name}_input_video.mp4"
                    save_video_clip_with_bbox(rotated_video_input, output_file, base_azimuth, elevation, event_class)
                    print(f"  Processed input video -> {output_file}")

        return high_quality_events
        
    except Exception as e:
        print(f"[PID:{os.getpid()}] Error processing folder {folder_path}: {e}")
        traceback.print_exc()
        return []

def process_event_folder_wrapper(args):
    """Wrapper function for multiprocessing"""
    event_folder, sofa_path, subfolder_output = args
    return process_audio_folder(event_folder, sofa_path, subfolder_output)

def main():
    parser = argparse.ArgumentParser(description="Process spatial audio files with augmentation")
    parser.add_argument("--input_dir", required=True, help="Root directory containing audio folders")
    parser.add_argument("--sofa", required=True, help="Path to SOFA file for spatial processing")
    parser.add_argument("--output_dir", required=True, help="Output directory for augmented files")
    parser.add_argument("--processes", type=int, default=None, help="Number of parallel processes (default: CPU count)")
     
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not os.path.exists(args.sofa):
        print(f"Error: SOFA file not found: {args.sofa}")
        return
    
    # Determine number of processes
    num_processes = args.processes or cpu_count()
    print(f"Using {num_processes} processes for parallel processing")
    
    all_high_quality_events = []
    
    # 收集所有需要处理的事件文件夹
    print("Scanning for event folders...")
    all_event_folders = []
    
    # for subfolder in input_dir.iterdir():
    for subfolder in input_dir.iterdir():
        if 'audio_choice' in subfolder.name:
            continue
        if subfolder.is_dir():
            for subsubfolder in subfolder.iterdir():
                # if 'sony' in subsubfolder.name:
                #     continue
                if subsubfolder.is_dir():
                    for event_folder in subsubfolder.iterdir():
                        if event_folder.is_dir():
                            subfolder_output = output_dir / subfolder.name / subsubfolder.name / event_folder.name
                            subfolder_output.mkdir(parents=True, exist_ok=True)
                            all_event_folders.append((event_folder, args.sofa, subfolder_output))
    
    print(f"Found {len(all_event_folders)} event folders to process")
    
    if all_event_folders:
        # 使用 imap 获得实时进度显示
        with Pool(processes=num_processes) as pool:
            try:
                with tqdm(total=len(all_event_folders), desc="Processing events", unit="folder") as pbar:
                    results = []
                    for result in pool.imap(process_event_folder_wrapper, all_event_folders):
                        results.append(result)
                        pbar.update(1)
                
                # 收集所有高质量事件
                for high_quality_events in results:
                    if high_quality_events:
                        all_high_quality_events.extend(high_quality_events)
                        
            except Exception as e:
                print(f"Error in multiprocessing pool: {e}")
                traceback.print_exc()
    
    # Write high quality events to file
    output_file = output_dir / "high_quality.txt"
    with open(output_file, 'w') as f:
        for event in sorted(set(all_high_quality_events)):
            f.write(f"{event}\n")
    
    print(f"\nProcessing complete!")
    print(f"High quality events: {len(set(all_high_quality_events))}")
    print(f"Results written to: {output_file}")
if __name__ == "__main__":
    main()