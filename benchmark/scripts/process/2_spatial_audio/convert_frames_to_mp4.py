import os
import cv2
import glob
import re
import numpy as np

def frames_to_mp4(frames_folder, output_path, fps=2.0):
    """
    Convert a sequence of frame images to MP4 video
    
    Args:
        frames_folder: Path to folder containing frame images
        output_path: Path for output MP4 file
        fps: Frames per second for output video (default: 30)
    """
    # Get all jpg files in the folder and sort them naturally
    frame_files = glob.glob(os.path.join(frames_folder, "*.jpg"))
    
    # Natural sort to handle frame_0.jpg, frame_1.jpg, ..., frame_10.jpg correctly
    def natural_sort_key(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]
    
    frame_files.sort(key=natural_sort_key)
    
    if not frame_files:
        print(f"No JPG files found in {frames_folder}")
        return False
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"Could not read first frame: {frame_files[0]}")
        return False
    
    height, width, layers = first_frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Converting {len(frame_files)} frames to {output_path}")
    
    # Process each frame
    for i, frame_path in enumerate(frame_files):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
        
        video_writer.write(frame)
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(frame_files)} frames")
    
    # Release everything
    video_writer.release()
    print(f"Video saved to: {output_path}")
    return True

def concat_frames_to_image(frames_folder, output_path, max_cols=5):
    """
    Concatenate all frames into a single grid image
    
    Args:
        frames_folder: Path to folder containing frame images
        output_path: Path for output concatenated image
        max_cols: Maximum number of columns in the grid (default: 5)
    """
    # Get all jpg files in the folder and sort them naturally
    frame_files = glob.glob(os.path.join(frames_folder, "*.jpg"))
    
    # Natural sort to handle frame_0.jpg, frame_1.jpg, ..., frame_10.jpg correctly
    def natural_sort_key(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]
    
    frame_files.sort(key=natural_sort_key)
    
    if not frame_files:
        print(f"No JPG files found in {frames_folder}")
        return False
    
    print(f"Concatenating {len(frame_files)} frames to {output_path}")
    
    # Read all frames
    frames = []
    for frame_path in frame_files:
        frame = cv2.imread(frame_path)
        if frame is not None:
            frames.append(frame)
        else:
            print(f"Warning: Could not read frame {frame_path}")
    
    if not frames:
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
    
    if success:
        print(f"Concatenated image saved: {output_path} ({rows}x{cols} grid)")
        return True
    else:
        print(f"Failed to save concatenated image: {output_path}")
        return False

def convert_all_folders(base_path):
    """
    Convert all frame folders in urbansas_samples to MP4 files
    
    Args:
        base_path: Path to urbansas_samples folder
    """
    if not os.path.exists(base_path):
        print(f"Base path does not exist: {base_path}")
        return
    
    # Get all subdirectories that contain frames
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        
        if os.path.isdir(item_path):
            # Check if this folder contains frame files
            frame_files = glob.glob(os.path.join(item_path, "frame_*.jpg"))
            
            if frame_files:
                # Extract ID from folder name (assuming format like acevedo0103_00_0_0.00_4.00)
                folder_name = os.path.basename(item_path)
                
                # Create output filenames
                mp4_filename = f"{folder_name}.mp4"
                mp4_output_path = os.path.join(base_path, mp4_filename)
                
                concat_filename = f"{folder_name}_frames.jpg"
                concat_output_path = os.path.join(base_path, concat_filename)
                
                print(f"\nProcessing folder: {folder_name}")
                
                # Convert to MP4
                mp4_success = frames_to_mp4(item_path, mp4_output_path, fps=30)
                
                # Create concatenated frames image
                concat_success = concat_frames_to_image(item_path, concat_output_path, max_cols=5)
                
                if mp4_success:
                    print(f"✓ Successfully created {mp4_filename}")
                else:
                    print(f"✗ Failed to create {mp4_filename}")
                    
                if concat_success:
                    print(f"✓ Successfully created {concat_filename}")
                else:
                    print(f"✗ Failed to create {concat_filename}")

# Main execution
if __name__ == "__main__":
    # Set your frames location here
    frames_location = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/urbansas_samples"  # Update this path as needed
    
    print("Starting frame to MP4 conversion and concatenation...")
    convert_all_folders(frames_location)
    print("\nConversion and concatenation complete!")