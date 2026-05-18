from math import e
import os
import re
import fireredtts
import torchaudio
from google import genai
from google.genai import types
from fireredtts import FireRedTTS
import json
import random
from tqdm import tqdm
from PIL import Image

os.environ['audioBench'] = "/home/xwang378/scratch/2025/AudioBench"
API = {}
with open(f"{os.environ['audioBench']}/.envs", "r") as f:
    env_vars = f.readlines()
for line in env_vars:
    name, value = line.strip().split('=')
    API[name] = value
client = genai.Client(api_key=API['Google_API_Key'])

def load_prompt(tts_dataset_dir):
    text_dir = os.path.join(tts_dataset_dir, "txt")
    audio_dir = os.path.join(tts_dataset_dir, "wav48_silence_trimmed")
    
    prompts = []
    for human_id in os.listdir(text_dir):
      # use 2nd text file
      text_id = os.listdir(os.path.join(text_dir, human_id))[1]
      text_path = os.path.join(text_dir, human_id, text_id)
      audio_path = os.path.join(audio_dir, human_id, f"{text_id.split('.')[0]}_mic1.flac")
      
      with open(text_path, "r") as f:
        text = f.read()
      prompts.append((text, audio_path))
    return prompts
  
def load_wenet_prompt(tts_dataset_dir):
  text_dir = os.path.join(tts_dataset_dir, "txts")
  audio_dir = os.path.join(tts_dataset_dir, "wavs")
  
  prompts = []
  for instance in os.listdir(text_dir):
    text_path = os.path.join(text_dir, instance)
    audio_path = os.path.join(audio_dir, instance.replace('.txt', '.wav'))
    with open(text_path, "r") as f:
      text = f.read()
    text = text.split("\t")[1].rstrip('\n')
    if len(text) <= 40 and len(text) >= 20:
      prompts.append((text, audio_path))
  
  return prompts

def convert_chinese_digits_with_gemini(text):
    if not re.search(r'\d', text):
        return text
    prompt = f"""
You are a helpful assistant that converts all Arabic numerals in Chinese sentences into proper Chinese numeral characters.

Your task:
- Convert all Arabic digits (e.g., 1, 2, 3) in the sentence into their Chinese numeral equivalents (e.g., 一, 二, 三).
- Pay special attention to the number “2”:
    - Use “两” instead of “二” when “2” is followed by a common measure word (like 个, 位, 只, 本, 条, 双, 辆, 名, 张, etc.)
    - But keep “二” in idiomatic expressions, formal honorifics, sequence words, or fixed terms (e.g., “您二位”, “第二个”, “二胡”, “二号线”)

Please return the modified sentence only.

Input: 我有2个苹果和3本书。
Output: 我有两个苹果和三本书。

Input: 明天早上7点开会。  
Output: 明天早上七点开会。

Input: 今天是2024年3月12日。  
Output: 今天是二零二四年三月十二日。

Now process the following sentence:
Input: {text}
Output: 
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt
    )

    return response.text.strip()

def replace_number_with_chinese(text):
  text = text.replace("1", "一")
  text = text.replace("2", "二")
  text = text.replace("3", "三")
  text = text.replace("4", "四")
  text = text.replace("5", "五")
  text = text.replace("6", "六")
  text = text.replace("7", "七")
  text = text.replace("8", "八")
  text = text.replace("9", "九")
  text = text.replace("0", "零")
  return text

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--local_id", type=int, default=0)
  args = parser.parse_args()
  local_id = args.local_id
  
  language = "en"

  # acoustic llm decoder
  tts = FireRedTTS(
          config_path="configs/config_24k.json",
          pretrained_path="/home/xwang378/scratch/2025/AudioBench/scripts/FireRedTTS/pretrained_models",
    )
  
  if language == "en":
    tts_dataset_dir = "/home/xwang378/scratch/2025/AudioBench/scripts/FireRedTTS/prompt"
    prompts = load_prompt(tts_dataset_dir)
    source_ocr_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/rendertext/hard"
  elif language == "zh":
    tts_dataset_dir = "/home/xwang378/scratch/2025/AudioBench/scripts/FireRedTTS/prompt/wenet/WenetSpeech4TTS_Premium_0"
    prompts = load_wenet_prompt(tts_dataset_dir)
    source_ocr_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/rendertext/trans_hard"
  else:
    raise ValueError(f"Language {language} not supported")

  all_source_ocr_files = [os.path.join(source_ocr_dir, f) for f in os.listdir(source_ocr_dir)]
  all_source_ocr_files = sorted(all_source_ocr_files)
  
  # start = local_id * 200
  # end = start + 200
  start = 0
  end = len(all_source_ocr_files)
  print(f"Processing from {start} to {end}")
  
  failed_txt = 'error.txt'
  for i in tqdm(range(start, end)):
    instance = all_source_ocr_files[i]
    random_prompt = random.choice(prompts)
    print(f"Processing {instance}")
    for choice in os.listdir(instance):
      choice_path = os.path.join(instance, choice)
      try:
        with open(choice_path, "r") as f:
          choice_text = f.read()
      except:
        print(f"Error reading {choice_path}")
        with open(failed_txt, "a") as f:
          f.write(f"{choice_path}\n")
        break
      
      out_wav_path = os.path.join(instance, choice_path.split('/')[-1].replace('.txt', '.wav'))
      
      if language == "zh":
        processed_text = convert_chinese_digits_with_gemini(choice_text)
        wav_output = tts.synthesize(
          prompt_wav=random_prompt[1],
          prompt_text=random_prompt[0],
          text=processed_text,
          lang=language,
          use_tn=False
        )
        
        if processed_text != choice_text:
          with open("gemini_response.txt", "a") as f:
            f.write(f"{choice_path}\n{choice_text}\n{processed_text}\n")
      else:
        wav_output = tts.synthesize(
          prompt_wav=random_prompt[1],
          prompt_text=random_prompt[0],
          text=choice_text,
          lang=language,
          use_tn=False
        )
        
      if wav_output is None:
        with open(failed_txt, "a") as f:
          f.write(f"{choice_path}\n")
        break
      
      wav_output = wav_output.detach().cpu()
      wav_output = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)(wav_output)
      torchaudio.save(out_wav_path, wav_output, 16000)
            
      
      
      
    
      
