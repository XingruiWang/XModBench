import os
import sys

sys.path.append("/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/")

import re
import json
import random
import torch
from transformers import AutoProcessor, AutoModel, AutoConfig, AutoModelForCausalLM
from audioBench import AudioBench
import argparse
from google import genai
from google.genai import types
from tqdm import tqdm
from pydantic import BaseModel

print("Available GPUs:", torch.cuda.device_count())

# Load API keys
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
    
def load_questions(path):
    """Load questions from JSON file"""
    with open(path, 'r') as f:
        questions = json.load(f)
    return questions

def get_question(questions, index):
    """Extract question details at given index"""
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

def load_model_and_processor(model_path, load_audio_in_video=True, num_video_frames=128, audio_length="max_3600"):
    """
    Load OmniVince model and processor.
    
    Args:
        model_path: Path to the OmniVince model
        load_audio_in_video: Whether to load audio in video files
        num_video_frames: Number of video frames to use
        audio_length: Maximum audio length
    """
    print(f"Loading OmniVince model from: {model_path}")
    
    # Load config and model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Configure model and processor settings
    model.config.load_audio_in_video = load_audio_in_video
    processor.config.load_audio_in_video = load_audio_in_video
    
    if num_video_frames > 0:
        model.config.num_video_frames = num_video_frames
        processor.config.num_video_frames = num_video_frames
    
    if audio_length != -1:
        model.config.audio_chunk_length = audio_length
        processor.config.audio_chunk_length = audio_length
    
    # Get default generation config
    generation_config = model.default_generation_config
    generation_kwargs = {"max_new_tokens": 1024, "max_length": 99999999}
    generation_config.update(**generation_kwargs)
    
    return model, processor, generation_config

def reevaluate_model(answer):
    """Use Gemini to extract answer choice from verbose response"""
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

def run_model(model, processor, generation_config, conversation, reason=False):
    """
    Run OmniVince model inference.
    
    Args:
        model: OmniVince model
        processor: OmniVince processor
        generation_config: Generation configuration
        conversation: Conversation history
        reason: Whether to extract reasoning
    """
    # Apply chat template
    # conversation = [{
    #     "role": "user",
    #     "content": [
    #         {"type": "video", "video":video_path},
    #         {"type": "text", "text": "Assess the video, followed by a detailed description of it's video and audio contents."}
    #     ]
    # }]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor([text])
    
    # Generate response
    output_ids = model.generate(
        input_ids=inputs.input_ids,
        media=getattr(inputs, 'media', None),
        media_config=getattr(inputs, 'media_config', None),
        generation_config=generation_config,
    )
    
    # Decode the output
    generated_text = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    # Remove the input prompt from the generated text if present

    if not reason:
        # Extract answer from generated text
        match = re.search(r'([A-D])(?:\.|,|\s|$)', generated_text)
        
        if match:
            answer = match.group(1)
        else:
            answer = generated_text
            print(f"Response is not a valid answer: {answer}. Re-running with reasoning extraction.")
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents = [
                    "The following response answers a multiple-choice question (A, B, C, or D), but includes additional words. "
                    "Please extract only the final answer choice (A, B, C, or D). If there is no answer choice, please return a random answer choice.",
                    generated_text
                ]
            ) 
            answer = response.text.strip().upper()
        return answer, None
    else:
        print(f"Re-running with reasoning extraction.")
        return "None", None
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents = [
                "The following response answers a multiple-choice question (A, B, C, or D) and the reasoning is included. ",
                "Please extract only the final answer choice (A, B, C, or D) and a detailed reasoning in a json format like this: {\"answer\": \"A\", \"reasoning\": \"...\"}.",
                generated_text
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": AnswerSchema,
            }
        ) 
        answer = response.parsed.answer.strip().upper()
        reasoning = response.parsed.reasoning.strip()

        return answer, reasoning

# ==================== 2. Generate conversation for each question ====================

def process_media_info(media_type, media_path):
    """Process media information based on type"""
    if media_type == 'video':
        return {
            "type": "video",
            "video": media_path
        }
    elif media_type == 'image':
        return {
            "type": "image",
            "image": media_path
        }
    elif media_type == 'audio':
        return {
            "type": "audio",
            "audio": media_path
        }
    else:  # text
        return {
            "type": "text",
            "text": media_path
        }

def generate_conversation(question, reason=False):
    """Generate conversation format for OmniVince"""
    condition_modality = question['condition_modality'].lower()
    choises_type = question['choices_type'].lower()
    
    # Build user content
    user_content = []
    
    # Add the question text
    user_content.append({"type": "text", "text": question['question']})
    
    # Add condition with label
    user_content.append({"type": "text", "text": f"{question['condition_modality']}:"})
    if condition_modality != 'text':
        user_content.append(process_media_info(condition_modality, question['condition_path']))
    else:
        user_content.append({"type": "text", "text": question['condition_path']})
    
    # Add choices
    choice_labels = ['A', 'B', 'C', 'D']
    for i, label in enumerate(choice_labels):
        user_content.append({"type": "text", "text": f"{label}:"})
        if choises_type != 'text':
            user_content.append(process_media_info(choises_type, question['choices_paths'][i]))
        else:
            user_content.append({"type": "text", "text": question['choices_paths'][i]})
    
    # Add instruction
    if not reason:
        user_content.append({"type": "text", "text": "Give the letter of the correct answer (A, B, C, or D)."})
    else:
        user_content.append({
            "type": "text", 
            "text": 'Please provide a detailed reasoning and then give the letter of the correct answer (A, B, C, or D) in a json format like this: {\"answer\": \"A\", \"reasoning\": \"...\"}.'
        })
    
    return [
        {
            "role": "user",
            "content": user_content
        }
    ]

# ==================== 3. Main evaluation loop ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', help='Root directory of AudioBench tasks', default='/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/')
    parser.add_argument('--task_name', help='Name of the task', default='perception/vggss_audio_vision')
    parser.add_argument('--sample', type=int, help='Number of samples to run', default=1)
    parser.add_argument('--save_dir', help='Directory to save results', default='/home/xwang378/scratch/2025/AudioBench/benchmark/results/')
    parser.add_argument('--reason', type=bool, default=False, help='Whether to run the reason script')
    parser.add_argument('--model', help='Model name', default='omnivince')
    parser.add_argument('--load_audio_in_video', type=bool, default=True, help='Whether to load audio in video')
    parser.add_argument('--num_video_frames', type=int, default=128, help='Number of video frames to use')
    parser.add_argument('--audio_length', type=str, default='max_3600', help='Maximum audio length')
    args = parser.parse_args()
    
    if args.reason:
        print(f"Running with reasoning")
    
    audiobench = AudioBench(root_dir=args.root_dir)

    task_name = args.task_name
    task_path = audiobench(task_name)
    questions = load_questions(task_path)
    
    # Load OmniVince model and processor
    model, processor, generation_config = load_model_and_processor(
        f"{os.environ['audioBench']}/models/OmniVinci",
        load_audio_in_video=args.load_audio_in_video,
        num_video_frames=args.num_video_frames,
        audio_length=args.audio_length
    )

    correct_count = 0
    all_count = 0
    format_error_count = 0
    run_error_count = 0
    save_result = {}
    save_result['task_name'] = task_name
    save_result['model'] = args.model
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
    hard_case_path = f"/home/xwang378/scratch/2025/AudioBench/benchmark/results/{args.model}/hard_case.json"
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
                text, reasoning = run_model(
                    model, processor, generation_config, conversation,
                    reason=args.reason
                )
            
            except Exception as e:
                print(f"Error generating conversation: {e}")
                text = f'Error: {e}'
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
    
    # Save results
    if not args.reason:
        save_path = os.path.join(args.save_dir, args.model, f"{task_name.replace('/', '_')}.json")
    else:
        save_path = os.path.join(args.save_dir, args.model, f"{task_name.replace('/', '_')}_reason.json")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(save_result, f, indent=4)
    print(f"Results saved to {save_path}")