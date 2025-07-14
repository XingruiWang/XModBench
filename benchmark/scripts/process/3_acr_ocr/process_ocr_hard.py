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
    
def generate_four_choices(text):
    
    instruction = f"""
        You are a helpful assistant that generates three **confusing sentence variants** of a given input sentence.

        Each output sentence should:
        - Maintain similar length, syntax, and tone as the original.
        - Contain a **different type of misunderstanding**, following these rules:

        1. One sentence must include **two or more key words that are changed into similar-sounding but different words**. These may differ by a vowel (a/e/i/o/u) or a similar-sounding consonant, but **should not be identical homophones**. The meaning must change clearly.

        2. Two sentences should use **semantic confusion**: change the meaning subtly by using near-synonyms, reversing cause-effect, or changing scope or intent.

        All outputs must remain **grammatically correct**, **stylistically similar**, and **potentially misleading** if read quickly.

        Your output must be a **JSON list of exactly 3 misleading variants**, with no extra explanation.

        Below are some in-context examples:

        ---
        """

    prompt = f"""
            Input Sentence:  
            "The 1966 Broadway musical 'Walking Happy' is based on the play."

            Output:
            [
            "The 1966 Broadway musical 'Walking Happy' is based on a screenplay.",            // semantic shift  
            "The 1966 Broadway musical 'Walking Happy' was the basis for a stage show.",      // reversed causality  
            "The 1966 Broadway musical 'Walking Happy' is based on the prey."                 // homophone (play → prey)
            ]

            ---
            Input Sentence:  
            "She feels very wealthy and decides to buy a new house."

            Output:
            [
            "She feels quite fortunate and chooses to build a new home.",                     // near-meaning, but misleading  
            "She thinks they need a larger home and begins looking at mansions.",             // exaggerated intention  
            "She feels very welty and decides to buy a new house."                            // homophone (wealthy → welty)
            ]

            ---
            Input Sentence:  
            "{text}"

            Output:
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[instruction, prompt]
    )
    
    return response.text
    


def main():
    root = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/rendertext/audio"
    output_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/rendertext/hard"
    
    all_instances = [os.path.join(root, file) for file in os.listdir(root) if file.endswith(".txt")]
    
    count = 0
    for instance in tqdm(all_instances[5:]):
        with open(instance, "r") as f:
            text = f.read()
        
        if len(text.split(" ")) < 10:
            continue
        
        edit_text = generate_four_choices(text)
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
            f.write(text)
        count += 1
        if count > 1000:
            break


if __name__ == "__main__":
    main()
    


    
    
    
    
    







