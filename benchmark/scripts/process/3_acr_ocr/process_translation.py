import os
from google import genai
import argparse
from tqdm import tqdm
# Load API Key from env file
API = {}
with open(f"{os.environ['audioBench']}/.envs", "r") as f:
    for line in f:
        name, value = line.strip().split('=')
        API[name] = value

client = genai.Client(api_key=API['Google_API_Key'])

def translate_text(english_text, target_language, model="gemini-2.0-flash"):
    prompt = f"Translate the following English text into {target_language}:\n\n\"{english_text}\n\n Output only the translated text, no other text."

    response = client.models.generate_content(
        model=model,
        contents=[prompt]
    )

    return response.text.strip()

def main(args):
    
    all_txt = [file for file in os.listdir(args.ocr_bench_path) if file.endswith(".txt")]
    all_txt = sorted(all_txt)
    
    for instance in tqdm(all_txt[:1000]):
        with open(os.path.join(args.ocr_bench_path, instance), "r") as f:
            text = f.read()
        translated = translate_text(text, args.language, model=args.model)
        
        os.makedirs(os.path.join(args.output_path, args.language), exist_ok=True)
        with open(os.path.join(args.output_path, args.language, instance), "w") as f:
            f.write(translated)

    
    print(f"\n[Translated to {args.language}]:\n{translated}")

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ocr_bench_path', type=str, required=True, help='Path to the OCR benchmark file')
    parser.add_argument('--language', type=str, required=True, help='Target language (e.g., Chinese, Spanish, French)')
    parser.add_argument('--model', type=str, default='gemini-2.0-pro', help='Gemini model to use (default: gemini-2.0-pro)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
    args = parser.parse_args()

    main(args)
