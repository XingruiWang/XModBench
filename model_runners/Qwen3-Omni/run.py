import os
import sys

sys.path.append("/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/")

import re
import json
import random
import torch
import soundfile as sf
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

from audioBench import AudioBench
import argparse
from google import genai
from google.genai import types

from tqdm import tqdm

print("Available GPUs:", torch.cuda.device_count())

API = {}
with open(f"{os.environ['audioBench']}/.envs", "r") as f:
    env_vars = f.readlines()
for line in env_vars:
    name, value = line.strip().split('=')
    API[name] = value
api_key = random.choice([API[key] for key in API if key.startswith('Google_API_Key')])
client = genai.Client(api_key=api_key)
from pydantic import BaseModel

class AnswerSchema(BaseModel):
    answer: str
    reasoning: str
    
def load_questions(path):
    with open(path, 'r') as f:
        questions = json.load(f)
    return questions

def get_question(questions, index):
    instance = questions[index]
    
    question = instance['question']
    
    condition_modality = instance['conditions']['modality']
    if condition_modality != 'Text':
        with open(instance['conditions']['input'], "rb") as f:
            condition_byte = f.read()
    else:
        condition_byte = instance['conditions']['input'] 
    
    condition_path = instance['conditions']['input']
    
    choices = instance['options']
    choises_type = instance['options']['A']['modality']
    choises_byte = {}
    
    for choice in choices:
        if choices[choice]['modality'] != 'Text':
            with open(choices[choice]['input'], "rb") as f:
                choises_byte[choice] = f.read()
        else:
            choises_byte[choice] = choices[choice]['input']
    
    choises_paths = [choices[choice]['input'] for choice in choices]
    correct_answer = instance['correct_answer']
    
    return {
        "question": question,
        "condition_path": condition_path,
        "condition_byte": condition_byte,
        'condition_modality': condition_modality,
        "choices_type": choises_type,
        "choices_paths": choises_paths,
        "choices_bytes": choises_byte,
        "correct_answer": correct_answer
    }
    
# ==================== 1. Main function: Load model and answer one question ====================

def load_model_and_processor(model_variant="instruct"):
    """
    Load Qwen3 model and processor.
    
    Args:
        model_variant: "instruct" or "thinking" to choose between the two Qwen3 variants
    """
    if model_variant == "instruct":
        MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    elif model_variant == "thinking":
        MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")
    
    print(f"Loading model: {MODEL_PATH}")
    
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    return model, processor

def reevaluate_model(answer):
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents = [
            "The following response answers a multiple-choice question (A, B, C, or D), but includes additional words. ",
            "Please extract only the final answer choice (A, B, C, or D).",
            answer
        ]
    ) 
    answer = response.text.strip().upper()
    return answer

def run_model(model, processor, conversation, use_audio_in_video=True, reason=False, speaker="Ethan"):
    """
    Run Qwen3 model inference.
    
    Args:
        model: Qwen3 model
        processor: Qwen3 processor
        conversation: Conversation history
        use_audio_in_video: Whether to use audio in video
        reason: Whether to extract reasoning
        speaker: Speaker voice to use for audio generation
    """
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video
    )
    inputs = inputs.to(model.device).to(model.dtype)

    # Generate with Qwen3 specific parameters
    text_ids, audio = model.generate(
        **inputs,
        thinker_return_dict_in_generate=True,
        use_audio_in_video=use_audio_in_video
    )
    
    # Decode only the generated text (excluding input prompt)
    text = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    if not reason:
        # Extract answer from generated text
        match = re.search(r'([A-D])(?:\.|,|\s|$)', text[0])
        
        if match:
            answer = match.group(1)
        else:
            answer = text[0]
            print(f"Response is not a valid answer: {answer}. Re-running with reasoning extraction.")
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents = [
                    "The following response answers a multiple-choice question (A, B, C, or D), but includes additional words. "
                    "Please extract only the final answer choice (A, B, C, or D). If there is no answer choice, please return a random answer choice.",
                    text[0]
                ]
            ) 
            answer = response.text.strip().upper()
        return answer, audio, None
    else:
        answer = text[0]
        print(f"Re-running with reasoning extraction.")
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents = [
                "The following response answers a multiple-choice question (A, B, C, or D) and the reasoning is included. ",
                "Please extract only the final answer choice (A, B, C, or D) and a detailed reasoning in a json format like this: {\"answer\": \"A\", \"reasoning\": \"...\"}.",
                text[0]
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": AnswerSchema,
            }
        ) 
        answer = response.parsed.answer.strip().upper()
        reasoning = response.parsed.reasoning.strip()

        return answer, audio, reasoning

# ==================== 2. Generate conversation for each image with fixed audio choices ====================

def process_media_info(media_type, media_path):
    if media_type == 'video':
        return {
            "type": "video",
            "video": media_path,
            "fps": 12,
            "max_frames": 12*5,
            "max_pixels": 512 * 512,
        }
    elif media_type == 'image':
        return {
            "type": "image",
            "image": media_path,
            "max_pixels": 512 * 512,
        }
    else:
        return {
            "type": media_type,
            media_type: media_path,
        }

def generate_conversation(question, reason=False):
    condition_modality = question['condition_modality'].lower()
    choises_type = question['choices_type'].lower()
    
    # Updated system prompt for Qwen3
    system_content = [
        {"type": "text", "text": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant capable of perceiving auditory and visual inputs, as well as generating text and speech."}
    ]
    
    if not reason:
        return [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question['question']},
                    {"type": "text", "text": f"{question['condition_modality']}:"},
                    process_media_info(condition_modality, question['condition_path']),
                    {"type": "text", "text": "A:"},
                    process_media_info(choises_type, question['choices_paths'][0]),
                    {"type": "text", "text": "B:"},
                    process_media_info(choises_type, question['choices_paths'][1]),
                    {"type": "text", "text": "C:"},
                    process_media_info(choises_type, question['choices_paths'][2]),
                    {"type": "text", "text": "D:"},
                    process_media_info(choises_type, question['choices_paths'][3]),
                    {"type": "text", "text": "Give the letter of the correct answer (A, B, C, or D)."}
                ]
            },
        ]
    else:
        return [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question['question']},
                    {"type": "text", "text": f"{question['condition_modality']}:"},
                    process_media_info(condition_modality, question['condition_path']),
                    {"type": "text", "text": "A:"},
                    process_media_info(choises_type, question['choices_paths'][0]),
                    {"type": "text", "text": "B:"},
                    process_media_info(choises_type, question['choices_paths'][1]),
                    {"type": "text", "text": "C:"},
                    process_media_info(choises_type, question['choices_paths'][2]),
                    {"type": "text", "text": "D:"},
                    process_media_info(choises_type, question['choices_paths'][3]),
                    {"type": "text", "text":'Please provide a detailed reasoning and then give the letter of the correct answer (A, B, C, or D) in a json format like this: {\"answer\": \"A\", \"reasoning\": \"...\"}.'}
                ]
            },
        ]

# ==================== 3. Example usage loop ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', help='Root directory of AudioBench tasks', default='/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/')
    parser.add_argument('--task_name', help='Name of the task', default='perception/vggss_audio_vision')
    parser.add_argument('--sample', type=int, help='Number of samples to run', default=1)
    parser.add_argument('--save_dir', help='Directory to save results', default='/home/xwang378/scratch/2025/AudioBench/benchmark/results/')
    parser.add_argument('--reason', type=bool, default=False, help='Whether to run the reason script')
    parser.add_argument('--model', help='Model name', default='qwen3_omni')
    parser.add_argument('--model_variant', choices=['instruct', 'thinking'], default='instruct', 
                        help='Which Qwen3 variant to use: instruct or thinking')
    parser.add_argument('--speaker', default='Ethan', help='Speaker voice for audio generation')
    args = parser.parse_args()
    
    if args.reason:
        print(f"Running with reasoning")
    
    audiobench = AudioBench(root_dir=args.root_dir)

    task_name = args.task_name
    task_path = audiobench(task_name)
    questions = load_questions(task_path)
    
    # Load Qwen3 model and processor
    model, processor = load_model_and_processor(model_variant=args.model_variant)

    correct_count = 0
    all_count = 0
    format_error_count = 0
    run_error_count = 0
    save_result = {}
    save_result['task_name'] = task_name
    save_result['model'] = f"qwen3_omni_{args.model_variant}"
    save_result['score'] = 0
    save_result['correct_count'] = 0
    save_result['all_count'] = 0
    
    save_result['results'] = {}

    # Handle hard cases if needed
    task_name2 = task_name.split('_')
    modality_name = '_'.join(task_name2[-2:])
    task_name2 = '_'.join(task_name2[:-2])
    
    # Optional: Load hard cases from previous runs
    hard_case_ids = []
    hard_case_path = f"/home/xwang378/scratch/2025/AudioBench/benchmark/results/qwen3_omni/hard_case.json"
    if os.path.exists(hard_case_path):
        with open(hard_case_path, "r") as f:
            hard_case = json.load(f)
        if task_name2 in hard_case:
            hard_case_ids = [int(id) for id in hard_case[task_name2].keys()]
    
    if args.sample == -1:
        sample_range = range(len(questions))
    else:
        if args.sample > len(questions):
            print(f"Sample size is greater than the number of questions, setting sample size to {len(questions)}")
            num_sample = len(questions)
            sample_range = range(len(questions))
        else:
            random.seed(42)
            num_sample = args.sample
            sample_range = sorted(random.sample(range(len(questions)), num_sample))
    
    # Optional: Filter to hard cases if needed
    if hard_case_ids:
        sample_range = [i for i in sample_range if i in hard_case_ids[:5]]
    
    for i in tqdm(sample_range):
        instance = get_question(questions, i)
        correct_answers = instance['correct_answer']
        
        previous_result = save_result['results'].get(str(i), None)
        if previous_result is None:
            try:
                conversation = generate_conversation(instance, reason=args.reason)
                text, audio, reasoning = run_model(
                    model, processor, conversation, 
                    use_audio_in_video=False,
                    reason=args.reason,
                    speaker=args.speaker
                )
            
            except Exception as e:
                print(f"Error generating conversation: {e}")
                text = f'Error: {e}'
                audio = None
                reasoning = None
        else:
            raise ValueError(f"Previous result already exists for {i}")
        
        if text.strip().upper() not in ['A', 'B', 'C', 'D']:
            text = reevaluate_model(text)

        print(f"GT: {correct_answers}, Answer: {text}, Reasoning: {reasoning if reasoning else 'N/A'}")
        
        if text == correct_answers:
            correct_count += 1
            is_correct = True
        else:
            is_correct = False

        if text.startswith('Error'):
            run_error_count += 1
        else:
            if text.strip().upper() not in ['A', 'B', 'C', 'D']:
                format_error_count += 1
                text = random.choice(['A', 'B', 'C', 'D'])
                print(f"Randomly selected answer: {text}")
                if text == correct_answers:
                    correct_count += 1
                    is_correct = True
            all_count += 1
            
        print(f"Accuracy: {correct_count / all_count * 100:.2f}. {correct_count}/{all_count}")
        print(f"Format error: {format_error_count}. Run error: {run_error_count}")
        
        save_result['results'][i] = {
            "question": instance['question'],
            "response": text,
            "correct_answer": correct_answers,
            "index": i,
            "is_correct": is_correct
        }
        if args.reason:
            save_result['results'][i]['reasoning'] = reasoning
    
    save_result['score'] = correct_count / all_count * 100 if all_count > 0 else 0
    save_result['correct_count'] = correct_count
    save_result['all_count'] = all_count
    save_result['format_error_count'] = format_error_count
    save_result['run_error_count'] = run_error_count
    
    # Save results with model variant in filename
    if not args.reason:
        save_path = os.path.join(args.save_dir, f"{task_name.replace('/', '_')}.json")
    else:
        save_path = os.path.join(args.save_dir, f"{task_name.replace('/', '_')}_reason.json")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(save_result, f, indent=4)
    print(f"Results saved to {save_path}")