import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback

video_root = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/Urbansas/video_2fps"
audio_root = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/Urbansas/audio"
output_root = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/urbansas_samples_videos_filtered"
os.makedirs(output_root, exist_ok=True)

csv_path = "/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/process/2_spatial_audio/movements/audio_with_motion_and_visibility_bounding_box.csv"

def uniform_sample(images_list, k=6):
    n = len(images_list)
    if n <= 8:
        return images_list  
    indices = np.linspace(0, n - 1, k).astype(int)
    return [images_list[i] for i in indices]

def extract_clip_audio_video(row_data, video_root, audio_root, output_root, fps=2):
    """Process a single row of data"""
    try:
        # Extract data from row
        clip_id = row_data["filename"]
        label = row_data["label"]
        start = float(row_data["start"])
        end = float(row_data["end"])
        class_id = row_data["class_id"]
        direction_x = float(row_data["direction_x"])
        direction_y = float(row_data["direction_y"])
        bounding_box = row_data["bounding_box"]
        
        video_path = os.path.join(video_root, f"{clip_id}.mp4")
        audio_path = os.path.join(audio_root, f"{clip_id}.wav")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(start * fps)
        end_frame = int(end * fps)
        
        if bounding_box and isinstance(bounding_box, str):
            try:
                bounding_box = eval(bounding_box)
            except:
                print(f"Failed to parse bounding_box for {clip_id}: {bounding_box}")
                bounding_box = []
        else:
            bounding_box = []
            
        visible_frame_time = [int(bbox[4]*2) for bbox in bounding_box]
        
        frame_list = []
        images_list = []
        output_dir = os.path.join(output_root, f"{clip_id}_{start:.2f}_{end:.2f}")
        os.makedirs(output_dir, exist_ok=True)
        
        for fid in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if ret:
                frame_list.append(fid)
                images_list.append(frame)
                
                # Draw bounding box if available
                if bounding_box and fid in visible_frame_time:
                    x, y, w, h, frame_time = bounding_box[visible_frame_time.index(fid)]
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                cv2.imwrite(os.path.join(output_dir, f"frame_{fid}.jpg"), frame)
                    
        cap.release()

        # Audio extraction
        audio_output_path = os.path.join(output_root, f"{clip_id}_{start:.2f}_{end:.2f}.wav")
        ffmpeg_cmd = (
            f"ffmpeg -y -i '{audio_path}' -ss {start} -to {end} -c copy "
            f"-loglevel quiet '{audio_output_path}'"
        )
        os.system(ffmpeg_cmd)

        return {
            "clip_id": clip_id,
            "class_id": class_id,
            "label": label,
            "start": start,
            "end": end,
            "direction_x": direction_x,
            "direction_y": direction_y,
            "audio_clip": audio_output_path,
            "video_clip": output_dir,
            "frames": frame_list
        }
        
    except Exception as e:
        print(f"Error processing {row_data.get('filename', 'unknown')}: {e}")
        traceback.print_exc()
        return None

def process_row_wrapper(args):
    """Wrapper function for multiprocessing"""
    row_data, video_root, audio_root, output_root = args
    return extract_clip_audio_video(row_data, video_root, audio_root, output_root)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process video clips in parallel")
    parser.add_argument("--processes", type=int, default=None, 
                       help="Number of parallel processes (default: CPU count)")
    args = parser.parse_args()
    
    # Load data
    spatial_audio_df = pd.read_csv(csv_path)
    
    # Determine number of processes
    num_processes = args.processes or cpu_count()
    print(f"Using {num_processes} processes for parallel processing")
    print(f"Processing {len(spatial_audio_df)} clips...")
    
    # Prepare arguments for multiprocessing
    tasks = []
    for _, row in spatial_audio_df.iterrows():
        tasks.append((row.to_dict(), video_root, audio_root, output_root))
    
    # Process in parallel
    results = []
    with Pool(processes=num_processes) as pool:
        try:
            with tqdm(total=len(tasks), desc="Processing clips", unit="clip") as pbar:
                for result in pool.imap(process_row_wrapper, tasks):
                    if result is not None:
                        results.append(result)
                    pbar.update(1)
        except Exception as e:
            print(f"Error in multiprocessing: {e}")
            traceback.print_exc()
    
    # Save results
    if results:
        output_df = pd.DataFrame(results)
        output_csv = "urbansas_extracted_samples.csv"
        output_df.to_csv(output_csv, index=False)
        print(f"Successfully processed {len(results)} clips")
        print(f"Results saved to: {output_csv}")
    else:
        print("No clips were successfully processed")

if __name__ == "__main__":
    main()