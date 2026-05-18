import os
import sys

sys.path.append("/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/")

import re
import json
import random
import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
from decord import VideoReader, cpu
from tqdm import tqdm

from audioBench import AudioBench
import argparse
from google import genai
from google.genai import types
from pydantic import BaseModel

print("Available GPUs:", torch.cuda.device_count())

# ==================== Image Processing Functions ====================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """Get frame indices for video sampling"""
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=8):
    """Load video and return pixel values and num_patches_list"""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

# ==================== Google API Setup ====================

API = {}
with open(f"{os.environ['audioBench']}/.envs", "r") as f:
    env_vars = f.readlines()
for line in env_vars:
    name, value = line.strip().split('=')
    API[name] = value
api_key = random.choice([API[key] for key in API if key.startswith('Google_API_Key')])
client = genai.Client(api_key=api_key)

class AnswerSchema(BaseModel):
    answer: str
    reasoning: str

# ==================== Question Loading Functions ====================

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
    choices_type = instance['options']['A']['modality']
    choices_byte = {}
    
    for choice in choices:
        if choices[choice]['modality'] != 'Text':
            with open(choices[choice]['input'], "rb") as f:
                choices_byte[choice] = f.read()
        else:
            choices_byte[choice] = choices[choice]['input']
    
    choices_paths = [choices[choice]['input'] for choice in choices]
    correct_answer = instance['correct_answer']
    
    return {
        "question": question,
        "condition_path": condition_path,
        "condition_byte": condition_byte,
        'condition_modality': condition_modality,
        "choices_type": choices_type,
        "choices_paths": choices_paths,
        "choices_bytes": choices_byte,
        "correct_answer": correct_answer
    }

# ==================== Model Loading and Inference ====================

def load_model_and_tokenizer(model_path='OpenGVLab/InternVL3_5-8B'):
    """Load InternVL model and tokenizer"""
    if torch.cuda.device_count() > 1:
        device_map = split_model(model_path)
    else:
        device_map = 'auto'
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    return model, tokenizer

def reevaluate_model(answer):
    """Use Gemini to extract answer from invalid response"""
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[
            "The following response answers a multiple-choice question (A, B, C, or D), but includes additional words. ",
            "Please extract only the final answer choice (A, B, C, or D). If there is no answer choice, please return a random answer choice.",
            answer
        ]
    ) 
    answer = response.text.strip().upper()
    return answer

def run_model(model, tokenizer, question_data, input_size=448, max_num=12, num_segments=8, reason=False):
    """Run InternVL model on a question with multi-modality support"""
    
    # Get modality types
    condition_modality = question_data['condition_modality'].lower()
    choices_type = question_data['choices_type'].lower()
    
    # Build the prompt
    if not reason:
        prompt = f"{question_data['question']}\n\n"
    else:
        prompt = f"{question_data['question']}\n\n"
    
    # Load condition based on modality
    pixel_values_list = []
    num_patches_list = []
    
    # Handle condition
    if condition_modality == 'text':
        prompt += f"{question_data['condition_modality']}:\n{question_data['condition_byte']}\n\n"
    elif condition_modality == 'image':
        prompt += f"{question_data['condition_modality']}:\n<image>\n\n"
        condition_pixel_values = load_image(question_data['condition_path'], input_size=input_size, max_num=max_num)
        pixel_values_list.append(condition_pixel_values)
        num_patches_list.append(condition_pixel_values.shape[0])
    elif condition_modality == 'video':
        # Load video frames
        condition_pixel_values, condition_num_patches = load_video(
            question_data['condition_path'], 
            input_size=input_size, 
            max_num=max_num,
            num_segments=num_segments
        )
        # Build video prefix with frame markers
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(condition_num_patches))])
        prompt += f"{question_data['condition_modality']}:\n{video_prefix}\n"
        pixel_values_list.append(condition_pixel_values)
        num_patches_list.extend(condition_num_patches)
    elif condition_modality == 'audio':
        # For audio, just include text description since InternVL doesn't process audio directly
        prompt += f"{question_data['condition_modality']}:\n[Audio file: {question_data['condition_path']}]\n\n"
    else:
        # Unknown modality, treat as text
        prompt += f"{question_data['condition_modality']}:\n{question_data.get('condition_byte', 'N/A')}\n\n"
    
    # Add choices
    for idx, choice_label in enumerate(['A', 'B', 'C', 'D']):
        if choices_type == 'text':
            prompt += f"{choice_label}: {question_data['choices_bytes'][choice_label]}\n"
        elif choices_type == 'image':
            prompt += f"{choice_label}:\n<image>\n"
            choice_pixel_values = load_image(question_data['choices_paths'][idx], input_size=input_size, max_num=max_num)
            pixel_values_list.append(choice_pixel_values)
            num_patches_list.append(choice_pixel_values.shape[0])
        elif choices_type == 'video':
            choice_pixel_values, choice_num_patches = load_video(
                question_data['choices_paths'][idx],
                input_size=input_size,
                max_num=max_num,
                num_segments=num_segments
            )
            choice_video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(choice_num_patches))])
            prompt += f"{choice_label}:\n{choice_video_prefix}"
            pixel_values_list.append(choice_pixel_values)
            num_patches_list.extend(choice_num_patches)
        elif choices_type == 'audio':
            prompt += f"{choice_label}: [Audio file: {question_data['choices_paths'][idx]}]\n"
        else:
            # Unknown type, use text
            prompt += f"{choice_label}: {question_data['choices_bytes'].get(choice_label, 'N/A')}\n"
    
    # Add final instruction
    if not reason:
        prompt += "\nGive the letter of the correct answer (A, B, C, or D)."
    else:
        prompt += '\nPlease provide a detailed reasoning and then give the letter of the correct answer (A, B, C, or D).'
    
    # Concatenate all pixel values if any exist
    if len(pixel_values_list) > 0:
        pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).cuda()
    else:
        pixel_values = None
    
    # Generate response
    generation_config = dict(max_new_tokens=1024 if reason else 128, do_sample=False)
    
    try:
        if pixel_values is not None:
            # If we have multiple patches from video, use num_patches_list
            if len(num_patches_list) > 1:
                response = model.chat(
                    tokenizer, 
                    pixel_values, 
                    prompt, 
                    generation_config,
                    num_patches_list=num_patches_list
                )
            else:
                response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        else:
            # Text-only question
            response = model.chat(tokenizer, None, prompt, generation_config)
        
        print(f"Raw response: {response}")
        
        if not reason:
            # Extract answer
            match = re.search(r'\b([A-D])\b', response)
            if match:
                answer = match.group(1)
            else:
                print(f"Response is not a valid answer: {response}. Re-running with reasoning extraction.")
                answer = reevaluate_model(response)
            return answer, None
        else:
            # Extract answer and reasoning using Gemini
            gemini_response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=[
                    "The following response answers a multiple-choice question (A, B, C, or D) and the reasoning is included. ",
                    "Please extract only the final answer choice (A, B, C, or D) and a detailed reasoning in a json format like this: {\"answer\": \"A\", \"reasoning\": \"...\"}.",
                    response
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": AnswerSchema,
                }
            )
            answer = gemini_response.parsed.answer.strip().upper()
            reasoning = gemini_response.parsed.reasoning.strip()
            return answer, reasoning
            
    except Exception as e:
        print(f"Error during model inference: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}", None

# ==================== Main Evaluation Loop ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', help='Root directory of AudioBench tasks', 
                        default='/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/')
    parser.add_argument('--task_name', help='Name of the task', 
                        default='perception/vggss_audio_vision')
    parser.add_argument('--sample', type=int, help='Number of samples to run', default=1)
    parser.add_argument('--save_dir', help='Directory to save results', 
                        default='/home/xwang378/scratch/2025/AudioBench/benchmark/results/')
    parser.add_argument('--reason', action='store_true', help='Whether to run with reasoning')
    parser.add_argument('--model', help='Model name', default='InternVL3_5-8B')
    parser.add_argument('--model_path', help='Model path', default='OpenGVLab/InternVL3_5-8B')
    parser.add_argument('--input_size', type=int, default=448, help='Input image size')
    parser.add_argument('--max_num', type=int, default=12, help='Max number of image tiles')
    parser.add_argument('--num_segments', type=int, default=8, help='Number of video frames to sample')
    args = parser.parse_args()
    
    if args.reason:
        print("Running with reasoning")
    
    # Load AudioBench
    audiobench = AudioBench(root_dir=args.root_dir)
    task_name = args.task_name
    task_path = audiobench(task_name)
    questions = load_questions(task_path)
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    print("Model loaded successfully!")
    
    # Initialize results tracking
    correct_count = 0
    all_count = 0
    format_error_count = 0
    run_error_count = 0
    save_result = {
        'task_name': task_name,
        'model': args.model,
        'score': 0,
        'correct_count': 0,
        'all_count': 0,
        'results': {}
    }
    
    # Determine sample range
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
    
    # Run evaluation
    for i in tqdm(sample_range):
        instance = get_question(questions, i)
        correct_answers = instance['correct_answer']
        
        try:
            text, reasoning = run_model(model, tokenizer, instance, 
                                       input_size=args.input_size, 
                                       max_num=args.max_num,
                                       num_segments=args.num_segments,
                                       reason=args.reason)
        except Exception as e:
            print(f"Error generating conversation: {e}")
            import traceback
            traceback.print_exc()
            text = f'Error: {e}'
            reasoning = None
        
        # Validate answer format
        if text.strip().upper() not in ['A', 'B', 'C', 'D']:
            if not text.startswith('Error'):
                text = reevaluate_model(text)
        
        print(f"GT: {correct_answers}, Answer: {text}, Reasoning: {reasoning}")
        
        # Check correctness
        is_correct = (text == correct_answers)
        if is_correct:
            correct_count += 1
        
        # Track errors
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
        
        print(f"Accuracy: {correct_count / all_count * 100:.2f}%. {correct_count}/{all_count}")
        print(f"Format error: {format_error_count}. Run error: {run_error_count}")
        
        # Save result
        save_result['results'][i] = {
            "question": instance['question'],
            "response": text,
            "correct_answer": correct_answers,
            "index": i,
            "is_correct": is_correct
        }
        if args.reason and reasoning:
            save_result['results'][i]['reasoning'] = reasoning
    
    # Finalize results
    save_result['score'] = correct_count / all_count * 100 if all_count > 0 else 0
    save_result['correct_count'] = correct_count
    save_result['all_count'] = all_count
    save_result['format_error_count'] = format_error_count
    save_result['run_error_count'] = run_error_count
    
    # Save results
    if not args.reason:
        save_path = os.path.join(args.save_dir, args.model, f"{task_name.replace('/', '_')}.json")
    else:
        save_path = os.path.join(args.save_dir, args.model, f"{task_name.replace('/', '_')}_reason.json")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(save_result, f, indent=4)
    print(f"\nResults saved to {save_path}")
    print(f"Final Accuracy: {save_result['score']:.2f}%")