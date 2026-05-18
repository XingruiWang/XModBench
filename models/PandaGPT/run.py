#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PandaGPT QA runner that follows EXACTLY the same input/output structure as the VITA script.
- Reuses: load_questions, get_question, generate_conversation signatures & return formats
- Keeps: main loop fields, result JSON schema, printouts, --reason behavior
- Swap backend: implements load_model_and_processor() and run_model() for PandaGPT

Usage:
CUDA_VISIBLE_DEVICES=0 python ./models/PandaGPT/run.py \
  --root_dir /home/xwang378/scratch/2025/AudioBench/benchmark/tasks/ \
  --task_name perception/general_audio_vision \
  --sample -1 \
  --save_dir /home/xwang378/scratch/2025/AudioBench/benchmark/results/ \
  --model_path /path/to/pandagpt/models \
  --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt \
  --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0 \
  --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt

Notes:
- PandaGPT supports multiple modalities: image, audio, video, thermal
- The script prints answer letter if detected (A-D). Otherwise returns raw text.
"""

import os
import sys
import re
import json
import argparse
import random
from typing import Any, Dict, List, Tuple, Optional

sys.path.append("/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/")
sys.path.append("/home/xwang378/scratch/2025/AudioBench/models/PandaGPT/code")

import torch
from tqdm import tqdm

from audioBench import AudioBench

# PandaGPT imports
from model.openllama import OpenLLAMAPEFTModel

print("Available GPUs:", torch.cuda.device_count())

# ==================== helpers matching VITA IO ====================

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


# ==================== PandaGPT Model Wrapper ====================

class PandaGPTInference:
    def __init__(self, model_args=None):
        """Initialize the PandaGPT model for inference"""
        if model_args is None:
            model_args = {
                'model': 'openllama_peft',
                'imagebind_ckpt_path': 'models/PandaGPT/pretrained_ckpt/imagebind_ckpt',
                'vicuna_ckpt_path': 'models/PandaGPT/pretrained_ckpt/vicuna_ckpt/7b_v0',
                'delta_ckpt_path': 'models/PandaGPT/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
                'stage': 2,
                'max_tgt_len': 128,
                'lora_r': 32,
                'lora_alpha': 32,
                'lora_dropout': 0.1,
            }
        
        print("Loading PandaGPT model...")
        self.model = OpenLLAMAPEFTModel(**model_args)
        
        # Load checkpoint
        delta_ckpt = torch.load(model_args['delta_ckpt_path'], map_location=torch.device('cpu'))
        self.model.load_state_dict(delta_ckpt, strict=False)
        self.model = self.model.eval().half().cuda()
        
        print("Model loaded successfully!")
        
        # Initialize conversation history
        self.history = []
        self.modality_cache = []
    
    def reset_conversation(self):
        """Reset conversation history and modality cache"""
        self.history = []
        self.modality_cache = []
    
    def prepare_prompt(self, input_text, history):
        """Prepare the prompt from conversation history"""
        prompt_text = ''
        for idx, (q, a) in enumerate(history):
            if idx == 0:
                prompt_text += f'{q}\n### Assistant: {a}\n###'
            else:
                prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
        
        if len(history) == 0:
            prompt_text += f'{input_text}'
        else:
            prompt_text += f' Human: {input_text}'
        
        return prompt_text
    
    def inference(self, 
                  text_input,
                  image_paths=None,
                  audio_paths=None,
                  video_paths=None,
                  thermal_paths=None,
                  max_length=256,
                  top_p=0.01,
                  temperature=1.0,
                  maintain_history=False):
        """
        Run inference on the model following PandaGPT's interface
        """
        
        # Check if any input data is provided
        if not text_input and not any([image_paths, audio_paths, video_paths, thermal_paths]):
            return "No input data provided! Please provide text input or upload media files."
        
        
        # Prepare prompt with history
        prompt_text = self.prepare_prompt(text_input, self.history)
        
        # Prepare file paths (filter out None values and convert to lists)
        image_paths = image_paths or []
        audio_paths = audio_paths or []
        video_paths = video_paths or []
        thermal_paths = thermal_paths or []
        
        # Generate response
        try:
            response = self.model.generate({
                'prompt': prompt_text,
                'image_paths': image_paths,
                'audio_paths': audio_paths,
                'video_paths': video_paths,
                'thermal_paths': thermal_paths,
                'top_p': top_p,
                'temperature': temperature,
                'max_tgt_len': max_length,
                'modality_embeds': self.modality_cache
            })
            print(f"Response: {response}")
            
            # Update history if maintaining conversation
            if maintain_history:
                self.history.append((text_input, response))
            
            return response
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return f"Error: {str(e)}"


# ==================== 1. Main function: Load model and answer one question ====================

def load_model_and_processor(
    imagebind_ckpt_path: str,
    vicuna_ckpt_path: str,
    delta_ckpt_path: str,
    model_path: str = None,
    stage: int = 2,
    max_tgt_len: int = 256,
    lora_r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1
):
    """Load PandaGPT model and return environment dict"""
    
    model_args = {
        'model': 'openllama_peft',
        'imagebind_ckpt_path': imagebind_ckpt_path,
        'vicuna_ckpt_path': vicuna_ckpt_path,
        'delta_ckpt_path': delta_ckpt_path,
        'stage': stage,
        'max_tgt_len': max_tgt_len,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
    }
    
    inferencer = PandaGPTInference(model_args)
    
    env = {
        'inferencer': inferencer,
        'max_tgt_len': max_tgt_len,
    }
    return env


def run_model(env: Dict[str, Any], conversation: List[Dict[str, Any]], use_audio_in_video: bool = False):
    """Run PandaGPT model with conversation input. Returns (answer_text, audio_bytes_or_None)."""
    
    inferencer = env['inferencer']
    max_tgt_len = env['max_tgt_len']
    
    # Parse conversation to extract text and media paths
    text_parts = []
    image_paths = []
    audio_paths = []
    video_paths = []
    thermal_paths = []
    
    # Extract system messages
    system_texts = []
    for msg in conversation:
        if msg.get('role') == 'system':
            for c in msg.get('content', []):
                if c.get('type') == 'text':
                    system_texts.append(c.get('text', ''))
    
    # Extract user message content
    user_msg = next((m for m in conversation if m.get('role') == 'user'), None)
    if user_msg is None:
        raise ValueError('No user message in conversation')
    
    for piece in user_msg.get('content', []):
        content_type = piece.get('type')
        
        if content_type == 'text':
            text_parts.append(piece.get('text', ''))
        elif content_type == 'image':
            path = piece.get('image') or piece.get('path')
            if path:
                image_paths.append(path)
        elif content_type == 'audio':
            path = piece.get('audio') or piece.get('path')
            if path:
                audio_paths.append(path)
        elif content_type == 'video':
            path = piece.get('video') or piece.get('path')
            if path:
                video_paths.append(path)
        elif content_type == 'thermal':
            path = piece.get('thermal') or piece.get('path')
            if path:
                thermal_paths.append(path)
    condition_modality = conversation[0]['content'][2]['type']
    choices_modality = conversation[0]['content'][4]['type']
    
    filtered_text_parts = []
    filtered_text_parts.append(text_parts[0]+'.')
    if condition_modality == 'text':
        filtered_text_parts.append(text_parts[2])
    if choices_modality == 'text':
        filtered_text_parts.extend(text_parts[2:-1])
    else:
        filtered_text_parts.append(f'A: the first {choices_modality}, B: the second {choices_modality}, C: the third {choices_modality}, D: the fourth {choices_modality}')
    filtered_text_parts.append('Please provide the letter of the correct answer (A, B, C, or D).')
    
    # Combine system prefix and user text
    
    system_prefix = ('\n'.join(system_texts).strip() + '\n') if system_texts else ''
    user_text = ' '.join(filtered_text_parts)
    full_text = system_prefix + user_text
    # Run inference
    try:
        response = inferencer.inference(
            text_input=full_text,
            image_paths=image_paths if image_paths else None,
            audio_paths=audio_paths if audio_paths else None,
            video_paths=video_paths if video_paths else None,
            thermal_paths=thermal_paths if thermal_paths else None,
            max_length=max_tgt_len,
            top_p=0.01,
            temperature=1.0,
            maintain_history=False
        )
        
        # Extract answer letter if present
        match = re.search(r'\b([A-D])\b', response)
        if match:
            answer = match.group(1)
        else:
            # answer = response.strip()
            print(f"No answer letter found in response: {response}")
            answer = random.choice(['A', 'B', 'C', 'D'])
            
        return answer, None
        
    except Exception as e:
        print(f"Error in PandaGPT inference: {e}")
        return f"Error: {str(e)}", None


# ==================== 2. Conversation template (unchanged from VITA code) ====================

def generate_conversation(question, reason=False):
    condition_modality = question['condition_modality'].lower()
    choises_type = question['choices_type'].lower()
    if not reason:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question['question']},
                    {"type": "text", "text": f"{question['condition_modality']}:"},
                    {"type": condition_modality, condition_modality: question['condition_path']},
                    {"type": "text", "text": "A:"},
                    {"type": choises_type, choises_type: question['choices_paths'][0]},
                    {"type": "text", "text": "B:"},
                    {"type": choises_type, choises_type: question['choices_paths'][1]},
                    {"type": "text", "text": "C:"},
                    {"type": choises_type, choises_type: question['choices_paths'][2]},
                    {"type": "text", "text": "D:"},
                    {"type": choises_type, choises_type: question['choices_paths'][3]},
                    {"type": "text", "text": "Give the letter of the correct answer (A, B, C, or D)."}
                ]
            },
        ]
    else:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question['question']},
                    {"type": "text", "text": f"{question['condition_modality']}:"},
                    {"type": condition_modality, condition_modality: question['condition_path']},
                    {"type": "text", "text": "A:"},
                    {"type": choises_type, choises_type: question['choices_paths'][0]},
                    {"type": "text", "text": "B:"},
                    {"type": choises_type, choises_type: question['choices_paths'][1]},
                    {"type": "text", "text": "C:"},
                    {"type": choises_type, choises_type: question['choices_paths'][2]},
                    {"type": "text", "text": "D:"},
                    {"type": choises_type, choises_type: question['choices_paths'][3]},
                    {"type": "text", "text": 'Please provide a detailed reasoning and then give the letter of the correct answer (A, B, C, or D) in a json format like this: {"answer": "A", "reasoning": "..."}.'}
                ]
            },
        ]


# ==================== 3. Main loop (kept identical) ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', help='Root directory of AudioBench tasks', default='/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/')
    parser.add_argument('--task_name', help='Name of the task', default='perception/general_audio_vision')
    parser.add_argument('--sample', type=int, help='Number of samples to run', default=1)
    parser.add_argument('--save_dir', help='Directory to save results', default='/home/xwang378/scratch/2025/AudioBench/benchmark/results/')
    parser.add_argument('--reason', type=bool, default=False, help='Whether to run the reason script')

    # PandaGPT-specific args
    parser.add_argument('--model_path', type=str, default=None, help='Path to PandaGPT model directory (optional)')
    parser.add_argument('--imagebind_ckpt_path', type=str, default='models/PandaGPT/pretrained_ckpt/imagebind_ckpt', help='Path to ImageBind checkpoint')
    parser.add_argument('--vicuna_ckpt_path', type=str, default='models/PandaGPT/pretrained_ckpt/vicuna_ckpt/7b_v0', help='Path to Vicuna checkpoint')
    parser.add_argument('--delta_ckpt_path', type=str, default='models/PandaGPT/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt', help='Path to PandaGPT delta checkpoint')
    parser.add_argument('--stage', type=int, default=2, help='Training stage')
    parser.add_argument('--max_tgt_len', type=int, default=256, help='Maximum generation length')
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')

    args = parser.parse_args()

    audiobench = AudioBench(root_dir=args.root_dir)

    task_name = args.task_name
    task_path = audiobench(task_name)
    questions = load_questions(task_path)

    # Load PandaGPT model + processors once
    env = load_model_and_processor(
        imagebind_ckpt_path=args.imagebind_ckpt_path,
        vicuna_ckpt_path=args.vicuna_ckpt_path,
        delta_ckpt_path=args.delta_ckpt_path,
        model_path=args.model_path,
        stage=args.stage,
        max_tgt_len=args.max_tgt_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    correct_count = 0
    all_count = 0
    save_result = {}
    save_result['task_name'] = task_name
    save_result['score'] = 0
    save_result['correct_count'] = 0
    save_result['all_count'] = 0
    save_result['error_count'] = 0

    save_result['results'] = {}

    if args.sample == -1:
        sample_range = range(len(questions))
    else:
        if len(questions) < args.sample:
            sample_range = range(len(questions))
        else:
            sample_range = range(args.sample)
        
    for i in tqdm(sample_range):
        instance = get_question(questions, i)
        correct_answers = instance['correct_answer']
            
        try:
            conversation = generate_conversation(instance, reason=args.reason)
            text, audio = run_model(env, conversation)
            
            if args.reason:
                # Try to parse JSON {answer, reasoning}
                try:
                    output = json.loads(text)
                    reasoning = output.get('reasoning', '')
                    text = output.get('answer', '')
                except Exception as e:
                    print(f"Error parsing reasoning: {e}")
                    reasoning = ''
                    # Keep text as-is (already a letter or raw string)
            print(f"GT: {correct_answers}, Answer: {text}")

            if text == correct_answers:
                correct_count += 1
                is_correct = True
            else:
                is_correct = False
            all_count += 1
        
        except Exception as e:
            print(f"Error generating conversation: {e}")
            text = f'Error: {e}'
            audio = None
            save_result['error_count'] += 1
            all_count += 1
            is_correct = False

        
        print(f"Accuracy: {correct_count / all_count * 100:.2f}. {correct_count}/{all_count}")
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

    if not args.reason:
        save_path = os.path.join(args.save_dir, f"{task_name.replace('/', '_')}.json")
    else:
        save_path = os.path.join(args.save_dir, f"{task_name.replace('/', '_')}_reason.json")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(save_result, f, indent=4)
    print(f"Results saved to {save_path}")