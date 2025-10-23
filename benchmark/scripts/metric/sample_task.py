import json
import os
from pathlib import Path
import shutil
import re

def copy_media_files(data, data_dir, base_input_dir):
    """
    Recursively find and copy media files referenced in the JSON data.
    Update the file paths in the data to point to the copied files.
    
    Args:
        data: JSON data (dict or list)
        data_dir (str): Directory to copy media files to
        base_input_dir (str): Base input directory to resolve relative paths
    
    Returns:
        Updated data with new file paths
    """
    media_extensions = {'.mp3', '.wav', '.mp4', '.avi', '.mov', '.jpg', '.jpeg', '.png', '.gif', '.flac', '.ogg', '.m4a'}
    copied_files = set()
    
    def process_value(value):
        if isinstance(value, str):
            # Check if this looks like a file path
            if any(ext in value.lower() for ext in media_extensions):
                # Try to find the actual file
                possible_paths = []
                
                # Try relative to base input dir
                if not os.path.isabs(value):
                    possible_paths.append(os.path.join(base_input_dir, value))
                else:
                    possible_paths.append(value)
                
                # Try removing leading slashes and joining with base dir
                clean_value = value.lstrip('/')
                possible_paths.append(os.path.join(base_input_dir, clean_value))
                
                for possible_path in possible_paths:
                    if os.path.exists(possible_path):
                        # File exists, copy it
                        file_name = os.path.basename(possible_path)
                        dest_path = os.path.join(data_dir, file_name)
                        
                        # Handle duplicate filenames
                        counter = 1
                        original_dest = dest_path
                        while dest_path in copied_files or os.path.exists(dest_path):
                            name, ext = os.path.splitext(original_dest)
                            dest_path = f"{name}_{counter}{ext}"
                            counter += 1
                        
                        try:
                            shutil.copy2(possible_path, dest_path)
                            copied_files.add(dest_path)
                            # Return relative path from tasks_sample directory
                            return f"data/{os.path.basename(dest_path)}"
                        except Exception as e:
                            print(f"Warning: Could not copy {possible_path}: {e}")
                            break
        
        return value
    
    def process_data(obj):
        if isinstance(obj, dict):
            return {key: process_data(process_value(value) if isinstance(value, str) else value) 
                   for key, value in obj.items()}
        elif isinstance(obj, list):
            return [process_data(process_value(item) if isinstance(item, str) else item) 
                   for item in obj]
        else:
            return process_value(obj) if isinstance(obj, str) else obj
    
    return process_data(data)

def sample_first_question_from_json(input_file_path, output_file_path, data_dir, base_input_dir):
    """
    Sample the first question from a JSON file and save to output path.
    Also copy any referenced media files.
    
    Args:
        input_file_path (str): Path to input JSON file
        output_file_path (str): Path to output JSON file
        data_dir (str): Directory to copy media files to
        base_input_dir (str): Base input directory for resolving relative paths
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different possible JSON structures
        sampled_data = None
        
        if isinstance(data, list) and len(data) > 0:
            # If data is a list, take the first item
            sampled_data = [data[0]]
        elif isinstance(data, dict):
            # If data is a dict, we need to check for common patterns
            if 'questions' in data and isinstance(data['questions'], list) and len(data['questions']) > 0:
                # Copy the structure but only keep the first question
                sampled_data = data.copy()
                sampled_data['questions'] = [data['questions'][0]]
            elif len(data) > 0:
                # If it's a dict with keys, take the first key-value pair
                first_key = list(data.keys())[0]
                if isinstance(data[first_key], list) and len(data[first_key]) > 0:
                    sampled_data = {first_key: [data[first_key][0]]}
                else:
                    sampled_data = {first_key: data[first_key]}
            else:
                sampled_data = data
        else:
            # If data is empty or not list/dict, keep as is
            sampled_data = data
        
        # Copy media files and update paths
        sampled_data = copy_media_files(sampled_data, data_dir, base_input_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # Write sampled data to output file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(sampled_data, f, indent=2, ensure_ascii=False)
        
        print(f"Sampled: {input_file_path} -> {output_file_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {input_file_path}: {str(e)}")
        return False

def sample_tasks(input_dir, output_dir):
    """
    Recursively find all JSON files in input_dir and sample first question from each.
    
    Args:
        input_dir (str): Input directory path
        output_dir (str): Output directory path
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    data_dir = output_path / "data"
    
    # Remove output directory if it exists and recreate it
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    json_files_processed = 0
    successful_samples = 0
    
    # Walk through all subdirectories and find JSON files
    for json_file in input_path.rglob("*.json"):
        # Create a simplified output filename
        # Extract meaningful parts from the path for naming
        relative_path = json_file.relative_to(input_path)
        path_parts = list(relative_path.parts)
        
        # Create a simplified name combining directory and file info
        if len(path_parts) >= 2:
            # Use category and subcategory info
            category = path_parts[0].replace('_', '')  # Remove underscores from category
            subcategory = path_parts[1] if len(path_parts) > 2 else ""
            filename = path_parts[-1]
            
            # Extract key info from filename
            filename_base = filename.replace('.json', '')
            # Remove common prefixes/suffixes to make it cleaner
            filename_base = re.sub(r'^(.*?)_questions_', '', filename_base)
            filename_base = re.sub(r'_audio_bench', '', filename_base)
            
            # Create simplified name
            if subcategory:
                output_filename = f"{category}_{subcategory}_{filename_base}.json"
            else:
                output_filename = f"{category}_{filename_base}.json"
        else:
            # Fallback to original filename
            output_filename = json_file.name
        
        # Clean up the filename
        output_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', output_filename)
        output_filename = re.sub(r'_+', '_', output_filename)  # Remove multiple underscores
        
        output_file_path = output_path / output_filename
        
        # Sample the first question and save
        if sample_first_question_from_json(str(json_file), str(output_file_path), str(data_dir), str(input_path)):
            successful_samples += 1
        json_files_processed += 1
    
    print(f"\nSampling complete!")
    print(f"Total JSON files processed: {json_files_processed}")
    print(f"Successful samples: {successful_samples}")
    print(f"Output directory: {output_dir}")
    print(f"Media files copied to: {data_dir}")

def main():
    # Define input and output directories
    input_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/tasks"
    output_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/tasks_sample"
    
    print(f"Starting task sampling...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Perform the sampling
    sample_tasks(input_dir, output_dir)

if __name__ == "__main__":
    main()
