import os
from google import genai
from google.genai import types
import time
from pathlib import Path

def check_file_structure_and_filter(base_dir):
    """
    Check file structure and use Gemini to validate matches
    Save low quality sample IDs to low.txt if ANY check fails
    """
    
    # Setup Gemini API
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: Please set GEMINI_API_KEY environment variable")
        return
    
    client = genai.Client(api_key=api_key)
    
    # Collect all sample IDs and corresponding files
    samples_data = {}
    low_quality_samples = []
    
    # Walk through all directories
    for root, dirs, files in os.walk(base_dir):
        # Skip root directory, only process sample ID directories
        if root == base_dir:
            continue
            
        sample_id = os.path.basename(root)
        
        # Initialize sample data
        if sample_id not in samples_data:
            samples_data[sample_id] = {
                'txt_files': [],
                'wav_files': [],
                'png_files': [],
                'path': root
            }
        
        # Categorize files
        for file in files:
            if file.endswith('.txt'):
                samples_data[sample_id]['txt_files'].append(file)
            elif file.endswith('.wav'):
                samples_data[sample_id]['wav_files'].append(file)
            elif file.endswith('.png'):
                samples_data[sample_id]['png_files'].append(file)
    
    print(f"Found {len(samples_data)} sample directories")
    
    # Check each sample
    for sample_id, data in samples_data.items():
        print(f"\nChecking sample: {sample_id}")
        
        sample_has_issues = False
        
        # Step 1: Check PNG and TXT matching
        print("  Step 1: Checking PNG and TXT matching...")
        if not check_png_txt_match(data):
            print("  ❌ PNG-TXT check failed")
            sample_has_issues = True
        else:
            print("  ✅ PNG-TXT check passed")
        
        # Step 2: Check Audio and TXT matching (only if Step 1 passed)
        if not sample_has_issues:
            print("  Step 2: Checking Audio and TXT matching with Gemini...")
            if not check_audio_txt_match(data, client):
                print("  ❌ Audio-TXT check failed")
                sample_has_issues = True
            else:
                print("  ✅ Audio-TXT check passed")
        else:
            print("  Step 2: Skipped due to Step 1 failure")
        
        # Save to low quality if any check failed
        if sample_has_issues:
            print(f"  🔴 Sample {sample_id} marked as low quality")
            low_quality_samples.append(sample_id)
        else:
            print(f"  🟢 Sample {sample_id} passed all checks")
    
    # Save low quality samples to file
    save_low_quality_samples(low_quality_samples)
    
    print(f"\n=== SUMMARY ===")
    print(f"Total samples: {len(samples_data)}")
    print(f"Low quality samples: {len(low_quality_samples)}")
    print(f"Good quality samples: {len(samples_data) - len(low_quality_samples)}")
    print(f"Results saved to low.txt")

def check_png_txt_match(sample_data):
    """
    Check if PNG and TXT files match by filename prefixes
    Returns True if all match, False if any mismatch
    """
    try:
        txt_files = sample_data['txt_files']
        png_files = sample_data['png_files']
        
        if not txt_files or not png_files:
            print("    Missing TXT or PNG files")
            return False
        
        # Extract prefixes
        txt_prefixes = set()
        for txt_file in txt_files:
            prefix = txt_file.replace('.txt', '')
            txt_prefixes.add(prefix)
        
        png_prefixes = set()
        for png_file in png_files:
            prefix = png_file.replace('.png', '')
            png_prefixes.add(prefix)
        
        # Check if sets match exactly
        if txt_prefixes == png_prefixes:
            print(f"    Found {len(txt_prefixes)} matching TXT-PNG pairs")
            return True
        else:
            missing_txt = png_prefixes - txt_prefixes
            missing_png = txt_prefixes - png_prefixes
            if missing_txt:
                print(f"    Missing TXT files for: {missing_txt}")
            if missing_png:
                print(f"    Missing PNG files for: {missing_png}")
            return False
            
    except Exception as e:
        print(f"    PNG-TXT check error: {e}")
        return False

def check_audio_txt_match(sample_data, client):
    """
    Check if Audio content matches TXT content using Gemini API
    Assumes file structure is already validated (same prefixes)
    Returns True if all content matches, False if any mismatch
    """
    try:
        txt_files = sample_data['txt_files']
        wav_files = sample_data['wav_files']
        
        if not txt_files or not wav_files:
            print("    Missing TXT or WAV files")
            return False
        
        print(f"    Checking content of {len(wav_files)} audio-text pairs with Gemini...")
        
        # Check each audio-text pair with Gemini (same prefix)
        for wav_file in wav_files:
            wav_prefix = wav_file.replace('.wav', '')
            txt_file = wav_prefix + '.txt'
            
            # Read text content
            txt_path = os.path.join(sample_data['path'], txt_file)
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
            except Exception as e:
                print(f"    Failed to read {txt_file}: {e}")
                return False
            
            if not text_content:
                print(f"    Empty text content in {txt_file}")
                return False
            
            # Check content with Gemini
            wav_path = os.path.join(sample_data['path'], wav_file)
            if not check_single_audio_text_with_gemini(client, wav_path, text_content, wav_file):
                return False  # If any single pair fails, entire sample fails
            
            # Add delay to avoid API rate limits
            time.sleep(1)
        
        print(f"    All {len(wav_files)} audio-text content pairs validated successfully")
        return True
        
    except Exception as e:
        print(f"    Audio-TXT content check error: {e}")
        return False

def check_single_audio_text_with_gemini(client, audio_path, text_description, audio_filename):
    """
    Check if single audio file matches text description using Gemini
    Returns True if matches, False if mismatches
    """
    try:
        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"    Audio file does not exist: {audio_path}")
            return False
        
        # Read audio data
        with open(audio_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        
        # Build prompt in English
        prompt = f"""Analyze this audio and determine if it matches: "{text_description}"
        
Please evaluate these aspects:
1. Does the speech content in the audio match the text description exactly?
2. Is the audio quality clear without noise or distortion?
3. Is the audio content complete and not truncated?
4. Are the language, tone, and emotion appropriate for the text?

Be STRICT in evaluation. Any quality issues or content mismatches should result in "NO".

Respond with only "YES" or "NO"."""
        
        # Call Gemini API using the exact script format
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, 
                      types.Part.from_bytes(mime_type="audio/wav", data=audio_data)
                      ]
        )
        
        # Parse response
        result = response.text.strip().upper()
        print(f"    {audio_filename}: Gemini -> {result}")
        
        return result == "YES"
        
    except Exception as e:
        print(f"    Gemini API failed for {audio_filename}: {e}")
        return False  # API call failed, mark as mismatch

def save_low_quality_samples(low_quality_samples):
    """Save low quality sample IDs to file"""
    try:
        with open('low_trans_hard.txt', 'w', encoding='utf-8') as f:
            for sample_id in low_quality_samples:
                f.write(f"{sample_id}\n")
        print(f"\nLow quality sample IDs saved to low.txt")
    except Exception as e:
        print(f"Failed to save file: {e}")

# Usage example
if __name__ == "__main__":
    # Set data directory path
    base_directory = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/audiobench_rendertext/trans_hard"  # Adjust path according to your file structure
    
    # Check environment variable
    if not os.getenv('GEMINI_API_KEY'):
        print("Please set GEMINI_API_KEY environment variable")
        print("Example: export GEMINI_API_KEY='your_api_key_here'")
    else:
        # Start checking and filtering
        check_file_structure_and_filter(base_directory)