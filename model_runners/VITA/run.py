#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VITA QA runner that follows EXACTLY the same input/output structure as your Qwen script.
- Reuses: load_questions, get_question, generate_conversation signatures & return formats
- Keeps: main loop fields, result JSON schema, printouts, --reason behavior
- Swap backend: implements load_model_and_processor() and run_model() for VITA

Usage:
CUDA_VISIBLE_DEVICES=0 python ./models/VITA/run.py \
  --root_dir /home/xwang378/scratch/2025/AudioBench/benchmark/tasks/ \
  --task_name perception/general_audio_vision \
  --sample -1 \
  --save_dir /home/xwang378/scratch/2025/AudioBench/benchmark/results/ \
  --model_path /home/xwang378/scratch/2025/AudioBench/models/VITA/VITA-1.5 \
  --model_type qwen2p5_instruct \
  --conv_mode qwen2p5_instruct 

Notes:
- VITA currently supports ONE audio segment per sample; if multiple appear we keep the first.
- choices modality may be Text/Image/Video/Audio; for Text we pass the string text.
- The script prints answer letter if detected (A-D). Otherwise returns raw text.
"""

import os
import sys
import re
import json
import argparse
from typing import Any, Dict, List, Tuple, Optional

sys.path.append("/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/")

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from audioBench import AudioBench
from torch.nn.utils.rnn import pad_sequence

# ---------------- VITA imports ----------------
try:
    from decord import VideoReader, cpu
except Exception:
    VideoReader, cpu = None, None

from vita.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,  # not directly used
    IMAGE_TOKEN_INDEX,
    MAX_IMAGE_LENGTH,
)
from vita.conversation import SeparatorStyle, conv_templates
from vita.model.builder import load_pretrained_model
from vita.util.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_audio_token,
    tokenizer_image_token,
)
from vita.util.utils import disable_torch_init

print("Available GPUs:", torch.cuda.device_count())

# ==================== helpers matching your IO ====================

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


# ==================== VITA preprocessing ====================

def _get_rawvideo_dec(
    video_path,
    image_processor,
    max_frames=MAX_IMAGE_LENGTH,
    min_frames=4,
    video_framerate=1,
    s=None,
    e=None,
    image_aspect_ratio="pad",
):
    if VideoReader is None:
        raise RuntimeError("decord is required for video processing but not available.")

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0.0 else 0.0
        end_time = end_time if end_time >= 0.0 else 0.0
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        raise FileNotFoundError(video_path)

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames <= 0:
        raise RuntimeError(f"No frames to decode in {video_path}")

    sample_fps = int(video_framerate)
    t_stride = int(round(float(fps) / sample_fps))
    all_pos = list(range(f_start, f_end + 1, t_stride))
    if len(all_pos) > max_frames:
        sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
    elif len(all_pos) < min_frames:
        sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)]
    else:
        sample_pos = all_pos

    patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

    # pad-to-square with mean color if configured
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    if image_aspect_ratio == "pad":
        patch_images = [
            expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean))
            for i in patch_images
        ]

    patch_images = [
        image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
        for i in patch_images
    ]

    patch_images = torch.stack(patch_images)
    slice_len = patch_images.shape[0]
    return patch_images, slice_len


# ==================== 1. Main function: Load model and answer one question ====================

def load_model_and_processor(model_path: str, model_base: Optional[str], model_type: str, conv_mode: str, frameCat: bool, video_fps: int):
    disable_torch_init()

    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, vision_tower, context_len = load_pretrained_model(
        model_path, model_base, model_name, model_type
    )

    model.resize_token_embeddings(len(tokenizer))

    vt = model.get_vision_tower()
    if not vt.is_loaded:
        vt.load_model()
    image_processor = vt.image_processor

    audio_encoder = model.get_audio_encoder()
    audio_encoder.to(dtype=torch.float16)
    audio_processor = audio_encoder.audio_processor

    if frameCat:
        from vita.util.data_utils_video_audio_neg_frameCat import dynamic_preprocess
    else:
        from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess

    env = {
        'tokenizer': tokenizer,
        'model': model,
        'image_processor': image_processor,
        'audio_processor': audio_processor,
        'dynamic_preprocess': dynamic_preprocess,
        'video_fps': video_fps,
        'conv_mode': conv_mode,
        'model_type': model_type,
    }
    return env


def run_model(env: Dict[str, Any], conversation: List[Dict[str, Any]], use_audio_in_video: bool = False):
    """Mimic Qwen's signature: returns (answer_text, audio_bytes_or_None)."""
    tokenizer = env['tokenizer']
    model = env['model']
    image_processor = env['image_processor']
    audio_processor = env['audio_processor']
    dynamic_preprocess = env['dynamic_preprocess']
    video_fps = env['video_fps']
    conv_mode = env['conv_mode']

    # Convert conversation to VITA prompt + tensors
    # 1) collect system prefix
    system_texts = []
    for msg in conversation:
        if msg.get('role') == 'system':
            for c in msg.get('content', []):
                if c.get('type') == 'text':
                    system_texts.append(c.get('text', ''))
    system_prefix = ('\n'.join(system_texts).strip() + '\n') if system_texts else ''

    # 2) user content to tokens and tensors
    user_msg = next((m for m in conversation if m.get('role') == 'user'), None)
    if user_msg is None:
        raise ValueError('No user message in conversation')

    prompt_parts: List[str] = []
    visual_chunks: List[torch.Tensor] = []
    audios: Optional[Dict[str, torch.Tensor]] = None

    def push_image_patches(patches: torch.Tensor, cnt: int):
        visual_chunks.append(patches)
        prompt_parts.append(DEFAULT_IMAGE_TOKEN * cnt)

    for piece in user_msg.get('content', []):
        t = piece.get('type')
        
        if t == 'text':
            prompt_parts.append(piece.get('text', ''))
        elif t == 'image':
            path = piece.get('image') or piece.get('path')
            img = Image.open(path).convert('RGB')
            ims, p_num = dynamic_preprocess(img, min_num=1, max_num=12, image_size=448, use_thumbnail=True)
            assert len(p_num) == 1
            patches = model.process_images(ims, model.config).to(dtype=model.dtype, device='cuda')
            push_image_patches(patches, p_num[0])
        elif t == 'video':
            path = piece.get('video') or piece.get('path')
            frames, slice_len = _get_rawvideo_dec(
                path, image_processor,
                max_frames=MAX_IMAGE_LENGTH,
                video_framerate=video_fps,
                s=piece.get('s'), e=piece.get('e'),
                image_aspect_ratio=getattr(model.config, 'image_aspect_ratio', None),
            )
            frames = frames.half().cuda()
            push_image_patches(frames, slice_len)
        elif t == 'audio':
            path = piece.get('audio') or piece.get('path')
            audio, audio_for_llm_lens = audio_processor.process(os.path.join(path))
            audio_length = audio.shape[0]
            audio = torch.unsqueeze(audio, dim=0)
            audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
            audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
            
            # import ipdb; ipdb.set_trace()
            if audios is None:
                audios = {
                    'audios': audio.half().cuda(),
                    'lengths': audio_length.half().cuda(),
                    'lengths_for_llm': audio_for_llm_lens.cuda(),
                }
            else:
                audios['audios'] = pad_sequence([audios['audios'][i] for i in range(len(audios['audios']))] + [audio.half().cuda().squeeze(0)], batch_first=True)
                audios['lengths'] = torch.cat([audios['lengths'], audio_length.half().cuda()], dim=0)
                audios['lengths_for_llm'] = torch.cat([audios['lengths_for_llm'], audio_for_llm_lens.cuda()], dim=0)
            # import ipdb; ipdb.set_trace()
           
            prompt_parts.append(DEFAULT_AUDIO_TOKEN)
            audio_added = True
        else:
            raise ValueError(f'Unsupported content type: {t}')
        
    # import ipdb; ipdb.set_trace()
    qs = "\n".join(prompt_parts)
    if system_prefix:
        qs = system_prefix + qs

    image_tensor = None
    if visual_chunks:
        image_tensor = torch.cat(visual_chunks, dim=0)

    # Build VITA conversation/prompt
    conv = conv_templates[conv_mode].copy()
    modality = 'image' if image_tensor is not None else 'lang'
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    
    # import ipdb; ipdb.set_trace()
    
    prompt = conv.get_prompt(modality)
    
    if image_tensor is None:
        image_tensor = torch.zeros((1, 3, 448, 448)).to(dtype=model.dtype, device="cuda")
        modality = "lang"
        
    if audios is not None:
        input_ids = tokenizer_image_audio_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    else:
        audio = torch.zeros(400, 80)
        audio_length = audio.shape[0]
        audio_for_llm_lens = 60
        audio = torch.unsqueeze(audio, dim=0)
        audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
        audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
        audios = dict()
        audios["audios"] = audio.half().cuda()
        audios["lengths"] = audio_length.half().cuda()
        audios["lengths_for_llm"] = audio_for_llm_lens.cuda()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            images=image_tensor,
            audios=audios,
            do_sample=False,
            temperature=0.01,
            top_p=None,
            num_beams=1,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            shared_v_pid_stride=None,
        )

    output_ids = out.sequences
    
    try:
        input_token_len = input_ids.shape[1]
        n_diff = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff > 0:
            output_ids = output_ids[:, input_token_len:]
    except Exception:
        pass

    raw_text = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
    text = raw_text.strip()
    if text.endswith(stop_str):
        text = text[:-len(stop_str)]
    text = text.strip()

    match = re.search(r'\b([A-D])\b', text)
    if match:
        answer = match.group(1)
    else:
        answer = text

    return answer, None  


# ==================== 2. Conversation template (unchanged from your code) ====================

def generate_conversation(question, reason=False):
    condition_modality = question['condition_modality'].lower()
    choises_type = question['choices_type'].lower()
    if not args.reason:
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

    # VITA-only args
    parser.add_argument('--model_path', type=str, default='/home/xwang378/scratch/2025/AudioBench/models/VITA/VITA-1.5', help='Path to VITA model directory')
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='qwen2p5_instruct')
    parser.add_argument('--conv_mode', type=str, default='qwen2p5_instruct')
    parser.add_argument('--frameCat', action='store_true')
    parser.add_argument('--video_fps', type=int, default=1)

    args = parser.parse_args()

    audiobench = AudioBench(root_dir=args.root_dir)

    task_name = args.task_name
    task_path = audiobench(task_name)
    questions = load_questions(task_path)

    # Load VITA model + processors once
    env = load_model_and_processor(
        model_path=args.model_path,
        model_base=args.model_base,
        model_type=args.model_type,
        conv_mode=args.conv_mode,
        frameCat=args.frameCat,
        video_fps=args.video_fps,
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
            continue

        
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
