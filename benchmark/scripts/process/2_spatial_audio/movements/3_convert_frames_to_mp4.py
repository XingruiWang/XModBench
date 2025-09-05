import os
import cv2
import glob
import re
import numpy as np
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import traceback

def frames_to_mp4(frames_folder, output_path, fps=2.0, quiet=True):
    """
    Convert a sequence of frame images to MP4 video with H.264 encoding
    
    Args:
        frames_folder: Path to folder containing frame images
        output_path: Path for output MP4 file
        fps: Frames per second for output video (default: 2.0)
        quiet: Whether to suppress output messages
    """
    # Get all jpg files in the folder and sort them naturally
    frame_files = glob.glob(os.path.join(frames_folder, "*.jpg"))
    
    # Natural sort to handle frame_0.jpg, frame_1.jpg, ..., frame_10.jpg correctly
    def natural_sort_key(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]
    
    frame_files.sort(key=natural_sort_key)
    
    if not frame_files:
        if not quiet:
            print(f"No JPG files found in {frames_folder}")
        return False
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        if not quiet:
            print(f"Could not read first frame: {frame_files[0]}")
        return False
    
    height, width, layers = first_frame.shape
    
    # Create temporary video file path
    output_path = Path(output_path)
    temp_path = output_path.parent / (output_path.stem + '_tmp.mp4')
    
    # Define the codec and create VideoWriter object for temporary video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
    
    if not quiet:
        print(f"Converting {len(frame_files)} frames to temporary video...")
    
    # Process each frame
    for i, frame_path in enumerate(frame_files):
        frame = cv2.imread(frame_path)
        if frame is None:
            if not quiet:
                print(f"Warning: Could not read frame {frame_path}")
            continue
        
        video_writer.write(frame)
    
    # Release the video writer
    video_writer.release()
    cv2.destroyAllWindows()
    
    if not quiet:
        print(f"Temporary mp4v video saved: {temp_path}")
    
    # Re-encode to H.264 using ffmpeg
    if not quiet:
        print(f"Re-encoding to H.264...")
    
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

def concat_frames_to_image(frames_folder, output_path, max_cols=5, quiet=True):
    """
    Concatenate all frames into a single grid image
    
    Args:
        frames_folder: Path to folder containing frame images
        output_path: Path for output concatenated image
        max_cols: Maximum number of columns in the grid (default: 5)
        quiet: Whether to suppress output messages
    """
    # Get all jpg files in the folder and sort them naturally
    frame_files = glob.glob(os.path.join(frames_folder, "*.jpg"))
    
    # Natural sort to handle frame_0.jpg, frame_1.jpg, ..., frame_10.jpg correctly
    def natural_sort_key(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]
    
    frame_files.sort(key=natural_sort_key)
    
    if not frame_files:
        if not quiet:
            print(f"No JPG files found in {frames_folder}")
        return False
    
    # Read all frames
    frames = []
    for frame_path in frame_files:
        frame = cv2.imread(frame_path)
        if frame is not None:
            frames.append(frame)
    
    if not frames:
        if not quiet:
            print("No valid frames found")
        return False
    
    # Calculate grid dimensions
    total_frames = len(frames)
    cols = min(max_cols, total_frames)
    rows = (total_frames + cols - 1) // cols  # Ceiling division
    
    # Get frame dimensions
    frame_height, frame_width = frames[0].shape[:2]
    
    # Create output image
    output_height = rows * frame_height
    output_width = cols * frame_width
    concat_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Place frames in grid
    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        
        y_start = row * frame_height
        y_end = y_start + frame_height
        x_start = col * frame_width
        x_end = x_start + frame_width
        
        concat_image[y_start:y_end, x_start:x_end] = frame
    
    # Save concatenated image
    success = cv2.imwrite(output_path, concat_image)
    
    if not quiet and success:
        print(f"Concatenated image saved: {output_path} ({rows}x{cols} grid)")
    
    return success

def process_single_folder(args):
    """Process a single folder (for multiprocessing)"""
    try:
        item_path, base_path = args
        folder_name = os.path.basename(item_path)
        
        # Check if this folder contains frame files
        frame_files = glob.glob(os.path.join(item_path, "frame_*.jpg"))
        
        if not frame_files:
            return {"folder": folder_name, "status": "no_frames", "mp4": False, "concat": False}
        
        # Create output filenames
        mp4_filename = f"{folder_name}.mp4"
        mp4_output_path = os.path.join(base_path, mp4_filename)
        
        if os.path.exists(mp4_output_path):
            return {"folder": folder_name, "status": "mp4_exists", "mp4": True, "concat": False}
        
        concat_filename = f"{folder_name}_frames.jpg"
        concat_output_path = os.path.join(base_path, concat_filename)
        
        # Convert to MP4
        mp4_success = frames_to_mp4(item_path, mp4_output_path, fps=2, quiet=True)
        
        # Create concatenated frames image
        concat_success = concat_frames_to_image(item_path, concat_output_path, max_cols=5, quiet=True)
        
        return {
            "folder": folder_name,
            "status": "processed",
            "mp4": mp4_success,
            "concat": concat_success,
            "mp4_path": mp4_output_path if mp4_success else None,
            "concat_path": concat_output_path if concat_success else None
        }
        
    except Exception as e:
        return {
            "folder": args[0] if args else "unknown",
            "status": "error",
            "error": str(e),
            "mp4": False,
            "concat": False
        }

def convert_all_folders_parallel(base_path, num_processes=None):
    """
    Convert all frame folders to MP4 files and concatenated images using parallel processing
    
    Args:
        base_path: Path to folder containing frame subfolders
        num_processes: Number of parallel processes (default: CPU count)
    """
    if not os.path.exists(base_path):
        print(f"Base path does not exist: {base_path}")
        return
    
    # Determine number of processes
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"Using {num_processes} processes for parallel processing")
    
    # Get all subdirectories that contain frames
    folders_to_process = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        
        if os.path.isdir(item_path):
            # Check if this folder contains frame files
            frame_files = glob.glob(os.path.join(item_path, "frame_*.jpg"))
            
            if frame_files:
                folders_to_process.append((item_path, base_path))
    
    if not folders_to_process:
        print("No folders with frame files found")
        return
    
    print(f"Found {len(folders_to_process)} folders to process")
    
    # Process folders in parallel
    results = []
    with Pool(processes=num_processes) as pool:
        try:
            with tqdm(total=len(folders_to_process), desc="Processing folders", unit="folder") as pbar:
                for result in pool.imap(process_single_folder, folders_to_process):
                    results.append(result)
                    pbar.update(1)
                    
        except Exception as e:
            print(f"Error in multiprocessing: {e}")
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    
    successful_mp4 = 0
    successful_concat = 0
    failed_folders = []
    no_frame_folders = []
    
    for result in results:
        if result["status"] == "processed":
            if result["mp4"]:
                successful_mp4 += 1
            if result["concat"]:
                successful_concat += 1
            if not result["mp4"] or not result["concat"]:
                failed_folders.append(result["folder"])
        elif result["status"] == "no_frames":
            no_frame_folders.append(result["folder"])
        elif result["status"] == "error":
            failed_folders.append(f"{result['folder']} (ERROR: {result.get('error', 'Unknown')})")
    
    print(f"Total folders processed: {len(results)}")
    print(f"Successful MP4 conversions: {successful_mp4}")
    print(f"Successful concatenations: {successful_concat}")
    print(f"Folders with no frames: {len(no_frame_folders)}")
    
    if failed_folders:
        print(f"\nFailed folders ({len(failed_folders)}):")
        for folder in failed_folders[:10]:  # Show first 10 failures
            print(f"  - {folder}")
        if len(failed_folders) > 10:
            print(f"  ... and {len(failed_folders) - 10} more")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert frame folders to MP4 and concatenated images")
    parser.add_argument("--base_path", default="/home/xwang378/scratch/2025/AudioBench/benchmark/Data/urbansas_samples_videos_filtered", help="Path to folder containing frame subfolders")
    parser.add_argument("--processes", type=int, default=8, 
                       help="Number of parallel processes (default: CPU count)")
    
    args = parser.parse_args()
    
    print("Starting parallel frame to MP4 conversion and concatenation...")
    convert_all_folders_parallel(args.base_path, args.processes)
    print("\nConversion and concatenation complete!")

# For backward compatibility
def convert_all_folders(base_path):
    """Legacy function - calls parallel version with default settings"""
    convert_all_folders_parallel(base_path)

if __name__ == "__main__":
    main()