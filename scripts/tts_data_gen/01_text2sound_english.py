import os
import fireredtts
import torchaudio
from fireredtts import FireRedTTS
import json
import random
from tqdm import tqdm
from PIL import Image

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


if __name__ == "__main__":

  # acoustic llm decoder
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--local_id', default=0, help='# function')
  args = parser.parse_args()

  tts = FireRedTTS(
          config_path="configs/config_24k.json",
          pretrained_path="/home/xwang378/scratch/2025/AudioBench/scripts/FireRedTTS/pretrained_models",
    )
  
  tts_dataset_dir = "/home/xwang378/scratch/2025/AudioBench/scripts/FireRedTTS/prompt"
  prompts = load_prompt(tts_dataset_dir)
  source_ocr_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/rendertext/unzipped"
  
  all_source_ocr_files = [os.path.join(source_ocr_dir, f) for f in os.listdir(source_ocr_dir) if f.endswith(".json")]
  all_source_ocr_files = sorted(all_source_ocr_files)
  
  local_id = int(args.local_id)
  start = local_id * 1000
  end = start + 1000
  for i in tqdm(range(start, end)):
    source_ocr_file = all_source_ocr_files[i]
    out_wav_path = os.path.join(f"/home/xwang378/scratch/2025/AudioBench/benchmark/Data/rendertext/audio/{source_ocr_file.split('/')[-1].replace('.json', '.wav')}")
    
    with open(source_ocr_file, "r") as f:
      source_ocr = json.load(f)
    text = ' '.join(source_ocr["ocr_annotation"]["text"])
    image_path = source_ocr_file.replace(".json", ".png")
    
    random_prompt = random.choice(prompts)
    
    wav_output = tts.synthesize(
      prompt_wav=random_prompt[1],
      prompt_text=random_prompt[0],
      text=text,
      lang="en",
      use_tn=False
    )
    
    if wav_output is None:
      continue
    
    # downsample to 16k
    wav_output = wav_output.detach().cpu()
    wav_output = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)(wav_output)
    torchaudio.save(out_wav_path, wav_output, 16000)
    
    with open(out_wav_path.replace(".wav", ".txt"), "w") as f:
      f.write(text)
    
    # image save and resize 
    image = Image.open(image_path)
    image = image.resize((512, 512))
    image.save(out_wav_path.replace(".wav", ".png"))
    
