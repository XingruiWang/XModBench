import os
from google import genai
from google.genai import types
import time
from tqdm import tqdm

def check_same_meaning_trans(sentence_1, sentence_2, client):
    """
    Check if single audio file matches text description using Gemini
    Returns True if matches, False if mismatches
    """
    try:
        # Build prompt in English
        prompt = f"""分析 "{sentence_1}" 和 "{sentence_2}" 两个句子，找出两个句子中所有不同的词语，并检查这些不同的词语是否具有完全相同的含义，比如“听起来好像发动机可能有点问题。”和“听起来好像引擎可能有点问题。”是相同的含义。
        如果两个句子具有完全相同的含义，则返回 "YES"，否则返回 "NO"。"""

        # Call Gemini API using the exact script format
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )
        
        # Parse response
        result = response.text.strip().upper()
        # print(f"    {sentence_1} and {sentence_2}: Gemini -> {result}")
        
        return result == "YES"
        
    except Exception as e:
        print(f"    Gemini API failed for {sentence_1} and {sentence_2}: {e}")
        return False  # API call failed, mark as mismatch

def filter_same_meaning_trans(base_dir):
    low_quality_audio_samples = "/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/process/3_acr_ocr/low_trans_hard.txt"
    with open(low_quality_audio_samples, "r") as f:
        low_quality_audio_samples = f.readlines()
    low_quality_audio_samples = [sample.strip() for sample in low_quality_audio_samples]
    
    client = genai.Client(api_key=os.getenv("Google_API_Key"))
    for instance_dir in tqdm(os.listdir(base_dir)):
        if instance_dir in low_quality_audio_samples:
            continue
        instance_dir = os.path.join(base_dir, instance_dir)
        original_trans_path = os.path.join(instance_dir, "original.txt")
        if not os.path.exists(original_trans_path):
            continue
        with open(original_trans_path, "r") as f:
            original_trans = f.read()
    
        for file in os.listdir(instance_dir):
            if file.endswith(".txt") and file != "original.txt":
                with open(os.path.join(instance_dir, file), "r") as f:
                    choice_content = f.read()
                    # if "发动机" in choice_content or "引擎" in choice_content:
                        # print(choice_content, file, instance_dir)
                    if check_same_meaning_trans(original_trans, choice_content, client):
                        low_quality_audio_samples.append(instance_dir)
                        print(f" {choice_content} is the same meaning as {original_trans}")
                        break
                    
    return low_quality_audio_samples

if __name__ == "__main__":
    base_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/audiobench_rendertext/trans_hard"
    low_quality_audio_samples = filter_same_meaning_trans(base_dir)
    with open("/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/process/3_acr_ocr/low_trans_hard.txt", "w") as f:
        for sample in low_quality_audio_samples:
            f.write(sample + "\n")