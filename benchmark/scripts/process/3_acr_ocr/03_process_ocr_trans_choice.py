import sys
sys.path.append("/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/")
import os
import json
from google import genai
from google.genai import types
import re
from audioBench import AudioBench
import argparse
from tqdm import tqdm
import ast
os.environ['audioBench'] = "/home/xwang378/scratch/2025/AudioBench"
API = {}
with open(f"{os.environ['audioBench']}/.envs", "r") as f:
    env_vars = f.readlines()
for line in env_vars:
    name, value = line.strip().split('=')
    API[name] = value
client = genai.Client(api_key=API['Google_API_Key'])
# Read image and audio

def parse_gpt_json_like_output(text):
    # Step 1: Remove markdown ```json ... ``` if exists
    text = re.sub(r"^```json\n|```$", "", text.strip())
    
    # Step 2: Remove inline comments (// ... or # ...)
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"#.*", "", text)
    
    # Optional: Remove any stray non-ASCII (like “→”) if you suspect weird encodings
    # text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Step 3: Use ast.literal_eval for safe evaluation
    try:
        return ast.literal_eval(text)
    except Exception as e:
        raise ValueError(f"Failed to parse cleaned output: {e}")
    
def generate_four_choices(correct_chinese):
        
    instruction = f"""
    You are a professional assistant for evaluating Chinese translations.

    Given a correct Chinese translation of an English sentence, generate **three incorrect translations** that are:

    1. Written in **fluent and natural Chinese**.
    2. Very close in **style, sentence structure, and length** to the correct version.
    3. Each output must contain a **real and noticeable translation error**, such as:
    - Semantic inaccuracy (e.g., wrong verb, mismatched subject)
    - Detail mistake (e.g., time, quantity, location)
    - Tone or emotional mismatch
    - Omission or addition of information
    - False cause-effect or logical drift

    Do **not** repeat the correct translation. All outputs must be **confusing at first glance**, but clearly wrong upon closer inspection.

    Return a **JSON list of exactly 3 incorrect Chinese sentences**. Do not include any explanation or formatting.

    Here are some examples:

    ---
    Correct Chinese Translation:  
    "我很自豪自己被选中为国家服务，你为此付出了巨大努力。"

    Output:
    [
    "我很自豪自己被选中为公司服务，你为此付出了巨大努力。",
    "我很高兴能服务于祖国，你其实没怎么努力。",
    "我很自豪被选为志愿者，你这方面没出什么力。"
    ]

    ---
    Correct Chinese Translation:  
    "请务必在周五早上之前交报告。这非常重要。"

    Output:
    [
    "请务必在周五下午之前交报告。这非常重要。",
    "请确认报告在周六之前完成。这个不太急。",
    "请在今天提交报告。我们马上就要用。"
    ]

    ---
    Correct Chinese Translation:  
    "{correct_chinese}"

    Output:
    """


    
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[instruction]
    )
    
    return response.text
    
def main():
    root = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/rendertext/audio"
    correct_trans_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/rendertext/translation/Chinese"
    output_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/rendertext/trans_hard"
    
    all_instances = [os.path.join(root, file) for file in os.listdir(correct_trans_dir) if file.endswith(".txt")]
    
    count = 0
    for instance in tqdm(all_instances[:]):
        with open(instance, "r") as f:
            text = f.read()
        
        with open(os.path.join(correct_trans_dir, instance.split("/")[-1].split(".txt")[0] + ".txt"), "r") as f:
            correct_trans = f.read()
        
        if len(text.split(" ")) < 10:
            continue
        
        edit_text = generate_four_choices(correct_trans)
        instance_name = instance.split("/")[-1].split(".txt")[0]
        
        try:
            edit_text = parse_gpt_json_like_output(edit_text)
        except:
            print(f"Error in parsing the output: {edit_text}")
            continue

        os.makedirs(os.path.join(output_dir, instance_name), exist_ok=True)
        for i, _edit_text in enumerate(edit_text):
            with open(os.path.join(output_dir, instance_name, f"choice_{i}.txt"), "w") as f:
                f.write(_edit_text)
        with open(os.path.join(output_dir, instance_name, "original.txt"), "w") as f:
            f.write(correct_trans)
        count += 1
        if count > 1000:
            break


if __name__ == "__main__":
    main()
    


    
    
    
    
    







