import sys
sys.path.append("/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/")
import os
import json
import base64
import random
from reka import ChatMessage, TypedMediaContent, TypedText
from reka.client import Reka

from google import genai

from audioBench import AudioBench
import argparse
from tqdm import tqdm
import cv2

API = {}
with open(f"{os.environ['audioBench']}/.envs", "r") as f:
    env_vars = f.readlines()
for line in env_vars:
    name, value = line.strip().split('=')
    API[name] = value

# Initialize Reka client
client = Reka(api_key=API['Reka_API_Key'])  # Update with your Reka API key name
google_client = genai.Client(api_key=API['Google_API_Key'])


def load_questions(path):
    with open(path, 'r') as f:
        questions = json.load(f)
    return questions

def resize_file(file_path, modality):
    if modality == 'Audio':
        return file_path
    elif modality == 'Image':
        scale = 224
        file_format = file_path.split('.')[-1]
        new_file_path = file_path.replace(f'.{file_format}', f'_224.{file_format}')
        if os.path.exists(new_file_path):
            return new_file_path
        image = cv2.imread(file_path)
        height, width = image.shape[:2]
        if height > width:
            new_height = scale
            new_width = int(width * new_height / height)
        else:
            new_width = scale
            new_height = int(height * new_width / width)
        image = cv2.resize(image, (new_width, new_height))

        cv2.imwrite(new_file_path, image)
        return new_file_path
    elif modality == 'Video':
        return file_path
    else:
        return file_path

def get_question(questions, index):
    instance = questions[index]
    
    question = instance['question']
    
    condition_modality = instance['conditions']['modality']
    
    if condition_modality != 'Text':
        resize_name = resize_file(instance['conditions']['input'], condition_modality)
        with open(resize_name, "rb") as f:
            condition_byte = f.read()
    else:
        condition_byte = instance['conditions']['input']
    
    choices = instance['options']
    choices_type = instance['options']['A']['modality']
    choices_byte = {}
    
    for choice in choices:
        if choices[choice]['modality'] != 'Text':
            resize_name = resize_file(choices[choice]['input'], choices[choice]['modality'])
            with open(resize_name, "rb") as f:
                choices_byte[choice] = f.read()
        else:
            choices_byte[choice] = choices[choice]['input']
    
    choices_paths = [choices[choice]['input'] for choice in choices]
    correct_answer = instance['correct_answer']
    
    return {
        "question": question,
        "condition_byte": condition_byte,
        'condition_modality': condition_modality,
        "choices_type": choices_type,
        "choices_paths": choices_paths,
        "choices_bytes": choices_byte,
        "correct_answer": correct_answer
    }

def encode_media_to_base64_url(media_bytes, media_type):
    """Convert media bytes to base64 data URL format for Reka"""
    base64_encoded = base64.b64encode(media_bytes).decode('utf-8')
    # Map media types to MIME types
    mime_type_map = {
        "Image": "image/jpeg",
        "Audio": "audio/wav",
        "Video": "video/mp4"
    }
    
    mime_type = mime_type_map[media_type]
    return f"data:{mime_type};base64,{base64_encoded}"

def _run_reka(instance, args):
    question = instance['question']
    condition_byte = instance['condition_byte']
    condition_modality = instance['condition_modality']
    
    choices_type = instance['choices_type']
    choices_bytes = instance['choices_bytes']
    
    # Build the content list for Reka
    content = []
    
    # Add the question text
    content.append({"type": "text", "text": question})
    
    # Add condition (if it's media)
    if condition_modality != 'Text':
        if condition_modality == 'Image':
            condition_url = encode_media_to_base64_url(condition_byte, condition_modality)
            content.append(TypedMediaContent(type="image_url", image_url=condition_url))
        elif condition_modality == 'Audio':
            # Note: Reka might not support audio directly, you may need to handle this differently
            condition_url = encode_media_to_base64_url(condition_byte, condition_modality)
            content.append(TypedMediaContent(type="audio_url", audio_url=condition_url))
        elif condition_modality == 'Video':
            # Note: Reka might not support video directly, you may need to handle this differently
            condition_url = encode_media_to_base64_url(condition_byte, condition_modality)
            content.append(TypedMediaContent(type="video_url", video_url=condition_url))
    else:
        content.append(TypedText(type="text", text=f"{condition_modality}: {condition_byte}"))
    # Add choices
    for choice_key in ['A', 'B', 'C', 'D']:
        content.append(TypedText(type="text", text=f"{choice_key}:"))
        
        if choices_type != 'Text':
            if choices_type == 'Image':
                choice_url = encode_media_to_base64_url(choices_bytes[choice_key], choices_type)
                content.append(TypedMediaContent(type="image_url", image_url=choice_url))
            elif choices_type == 'Audio':
                # Note: Handle audio appropriately based on Reka's capabilities
                choice_url = encode_media_to_base64_url(choices_bytes[choice_key], choices_type)
                content.append(TypedMediaContent(type="audio_url", audio_url=choice_url))
            elif choices_type == 'Video':
                # Note: Handle video appropriately based on Reka's capabilities
                choice_url = encode_media_to_base64_url(choices_bytes[choice_key], choices_type)
                content.append(TypedMediaContent(type="video_url", video_url=choice_url))
        else:
            content.append(TypedText(type="text", text=choices_bytes[choice_key]))
    
    if not args.reason:
        # Add instruction for simple answer
        content.append(TypedText(type="text", text="Give the letter of the correct answer (A, B, C, or D)."))
        
        try:
            response = client.chat.create(
                messages=[
                    ChatMessage(
                        content=content,
                        role="user",
                    )
                ],
                model=args.model,
            )
            
            response_text = response.responses[0].message.content
            
            # Check if response is valid
            if response_text.strip().upper() not in ['A', 'B', 'C', 'D']:
                print(f"Response is not a valid answer: {response_text.strip()}. Re-running with reasoning extraction.")
                
                # Try to extract answer from the response
                response = google_client.models.generate_content(
                    model="gemini-2.0-flash-lite",
                    # model="gemini-2.5-flash",
                    # model=args.model, # model="gemini-2.0-flash",  # or "gemini-2.5-flash"
                    contents = [
                        "The following response answers a multiple-choice question (A, B, C, or D), but includes additional reasoning. "
                        "Please extract only the final answer choice (A, B, C, or D).",
                        response_text
                    ]
                )
                response_text = response.text
            
            return response_text
            
        except Exception as e:
            print(f"Error in Reka API call: {e}")
            return None
    
    else:
        # Add instruction for detailed reasoning
        content.append(TypedText(type="text", text='Please provide a detailed reasoning and then give the letter of the correct answer (A, B, C, or D) in a json format like this: {"answer": "A", "reasoning": "..."}.'))
        
        try:
            response = client.chat.create(
                messages=[
                    ChatMessage(
                        content=content,
                        role="user",
                    )
                ],
                model=args.model,
            )
            
            response_text = response.responses[0].message.content
            
            try:
                output = eval(response_text.strip())
                answer = output.get('answer', '').strip().upper()
                reasoning = output.get('reasoning', '').strip()
            
            except Exception as e:
                print(f"Response is not a valid JSON: {response_text.strip()}. Re-running with reasoning extraction.")
                response = google_client.models.generate_content(
                    model="gemini-2.0-flash-lite",
                    # model="gemini-2.5-flash",
                    # model=args.model, # model="gemini-2.0-flash",  # or "gemini-2.5-flash"
                    contents = [
                        "For the following response, extract the final answer choice (A, B, C, or D) and the reasoning into a json format like this: {\"answer\": \"A\", \"reasoning\": \"...\"}.",
                        response_text
                    ]
                )  
                output = eval(response.text.strip())
                answer = output.get('answer', '').strip().upper()
                reasoning = output.get('reasoning', '').strip()

            return {
                'answer': answer,
                'reasoning': reasoning
            }
            
        except Exception as e:
            print(f"Error in Reka API call: {e}")
            return None

def run_reka(questions, index, args):
    instance = get_question(questions, index)
    
    try:
        response = _run_reka(instance, args)
        return response
    except Exception as e:
        print(f"Error processing question {index}: {e}")
        return None
    
def run_all_reka(task_name, questions, args, sample=100, save_dir=None):
    correct_count = 0
    all_count = 0
    save_result = {}
    save_result['task_name'] = task_name
    save_result['score'] = 0
    save_result['correct_count'] = 0
    save_result['all_count'] = 0
    
    save_result['results'] = {}
    
    if sample > len(questions):
        print(f"Sample is greater than the number of questions, setting sample to {len(questions)}")
        sample = range(len(questions))
    if sample < len(questions):
        sample = sorted(random.sample(range(len(questions)), sample))
    if sample == -1:
        sample = range(len(questions))
        
    for i in tqdm(sample):
        try:
            response = run_reka(questions, i, args)
            original_response = response
            reasoning = ''
            if response is not None:
                all_count += 1
                reasoning = ''
                if isinstance(response, dict):
                    reasoning = response.get('reasoning', '').strip()
                    response = response.get('answer', '').strip()
                if response.strip().upper() == questions[i]['correct_answer'].upper():
                    correct_count += 1
                    is_correct = True
                else:
                    if response.strip().upper() not in ['A', 'B', 'C', 'D']:
                        random_answer = random.choice(['A', 'B', 'C', 'D'])
                        response = random_answer
                    is_correct = response.strip().upper() == questions[i]['correct_answer'].upper()
                
                print(f"Question {i}: {response.strip()} (Correct: {questions[i]['correct_answer']}); Current Score: {correct_count}/{all_count} = {correct_count / all_count * 100:.5f}%")
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            response = random.choice(['A', 'B', 'C', 'D'])
            original_response = f"Error: {e}"
            reasoning = ''
            is_correct = response.strip().upper() == questions[i]['correct_answer'].upper()
        save_result['results'][i] = {
            "question": questions[i]['question'],
            "response": response.strip() if response else None,
            "original_response": original_response,
            'reasoning': reasoning,
            "correct_answer": questions[i]['correct_answer'],
            "index": i,
            "is_correct": is_correct if response else False
        }

    save_result['score'] = correct_count / all_count * 100 if all_count > 0 else 0
    save_result['correct_count'] = correct_count
    save_result['all_count'] = all_count
    
    if args.reason:
        save_path = f"{save_dir}/{task_name.replace('/', '_')}_reason_reka.json"
    else:
        save_path = f"{save_dir}/{task_name.replace('/', '_')}_reka.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(save_result, f, indent=4)
        
    print(f"Results saved to {save_path}")

def main(args):
    audiobench = AudioBench(root_dir=args.root_dir)

    task_name = args.task_name
    task_path = audiobench(task_name)
    questions = load_questions(task_path)

    run_all_reka(task_name, questions, args, sample=args.sample, save_dir=args.save_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', help='Root directory of AudioBench tasks', default='/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/')
    parser.add_argument('--task_name', help='Name of the task', default='perception/vggss_audio_vision')
    parser.add_argument('--sample', type=int, help='Number of samples to run', default=1)
    parser.add_argument('--save_dir', help='Directory to save results', default='/home/xwang378/scratch/2025/AudioBench/benchmark/results/')
    parser.add_argument('--model', help='Model to use for generation', default='reka-core-20240501')
    parser.add_argument('--reason', type=bool, default=False, help='Whether to run the reason script')
    args = parser.parse_args()
    main(args)