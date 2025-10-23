import sys
sys.path.append("/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/")
import os
import json
from google import genai
from google.genai import types

from audioBench import AudioBench
import argparse
from tqdm import tqdm
import random

API = {}
with open(f"{os.environ['audioBench']}/.envs", "r") as f:
    env_vars = f.readlines()
for line in env_vars:
    name, value = line.strip().split('=')
    API[name] = value

api_key = random.choice([API[key] for key in API if key.startswith('Google_API_Key')])
print(f"Using API key: {api_key}")
client = genai.Client(api_key=api_key)
# Read image and audio


from pydantic import BaseModel

class AnswerSchema(BaseModel):
    answer: str
    reasoning: str

    
def load_questions(path):
    with open(path, 'r') as f:
        questions = json.load(f)
    return questions

def load_dual_questions(path_at, path_vt):
    questions_at = load_questions(path_at)
    questions_vt = load_questions(path_vt)
    questions = []
    for question_at, question_vt in zip(questions_at, questions_vt):
        question_a = question_at['question']
        question_v = question_vt['question']
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            # model="gemini-2.5-flash",
            # model=args.model, # model="gemini-2.0-flash",  # or "gemini-2.5-flash"
            contents = [
                "The following is a pair of questions where the first question is about given an audio to find a text option, and the second question is about given a image or video to find an text option. "
                "Question 1:",
                question_at['question'],
                "Question 2:",
                question_vt['question'],
                "Please convert this two in one question, which is given an audio and a image or video together to find an text option. \n",
                "New Question:",
            ]
        )
        question = response.text
        question = {
            "question": question,
            "question_a": question_at['question'],
            "question_v": question_vt['question'],
            "conditions_a": question_at['conditions'],
            "conditions_v": question_vt['conditions'],
            "options": question_vt['options'],
            "correct_answer": question_at['correct_answer'],
        }
        questions.append(question)
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
        "condition_byte": condition_byte,
        'condition_modality': condition_modality,
        "choices_type": choises_type,
        "choices_paths": choises_paths,
        "choices_bytes": choises_byte,
        "correct_answer": correct_answer
    }
    
def get_question_audio_vision_text(questions, index):
    instance = questions[index]
    
    question = instance['question']
    
    with open(instance['conditions_a']['input'], "rb") as f:
        condition_byte_a = f.read()
    with open(instance['conditions_v']['input'], "rb") as f:
        condition_byte_v = f.read()

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
        "condition_byte_a": condition_byte_a,
        "condition_byte_v": condition_byte_v,
        'condition_modality_a': 'Audio',
        'condition_modality_v': 'Image',
        "choices_type": choises_type,
        "choices_paths": choises_paths,
        "choices_bytes": choises_byte,
        "correct_answer": correct_answer
    }
    
def _run_genimi(instance, args):
    
    modality_to_format = {
        "Image": "image/jpeg",
        "Audio": "audio/wav",
        "Text": "text"
    }
    question = instance['question']
    condition_byte = instance['condition_byte']
    condition_modality = instance['condition_modality']
    
    choices_type = instance['choices_type']
    choices_bytes = instance['choices_bytes']
    
    condition_format_str = modality_to_format.get(condition_modality, "text")
    choices_format_str = modality_to_format.get(choices_type, "text")
    
    if condition_modality == 'Text':
        condition_data = condition_byte
    elif condition_modality == 'Video':
        condition_data = types.Part(
                inline_data=types.Blob(data=condition_byte, mime_type='video/mp4')
            ),
    else:
        condition_data = types.Part.from_bytes(data=condition_byte, mime_type=condition_format_str)
    choise_data = {}
    
    for choice in choices_bytes:
        if choices_type == 'Text':
            choise_data[choice] = choices_bytes[choice]
        elif choices_type == 'Video':
            choise_data[choice] = types.Part(
                inline_data=types.Blob(data=choices_bytes[choice], mime_type='video/mp4')
            ),
        else:
            choise_data[choice] = types.Part.from_bytes(data=choices_bytes[choice], mime_type=choices_format_str),
    if not args.reason:
        contents = [
            question,
            f"{condition_modality}:",
            condition_data,
            "A:",
            choise_data['A'],
            "B:",
            choise_data['B'],
            "C:",
            choise_data['C'],
            "D:",
            choise_data['D'],
            "Give the letter of the correct answer (A, B, C, or D)."
        ]
        if args.model == "gemini-2.0-pro":
            print(f"Running model: gemini-1.5-pro")
            response = client.models.generate_content(
                model='gemini-1.5-pro',
                # model="gemini-1.5-pro",
                # model=args.model, # model="gemini-2.0-flash",  # or "gemini-2.5-flash"
                contents=contents
            )
        else:
            print(f"Running model: {args.model}")
            response = client.models.generate_content(
                model=args.model,
                # model="gemini-1.5-pro",
                # model=args.model, # model="gemini-2.0-flash",  # or "gemini-2.5-flash"
                contents=contents
            )
        
        if response.text.strip().upper() not in ['A', 'B', 'C', 'D']:
            print(f"Response is not a valid answer: {response.text.strip()}. Re-running with reasoning extraction.")
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                # model="gemini-2.5-flash",
                # model=args.model, # model="gemini-2.0-flash",  # or "gemini-2.5-flash"
                contents = [
                    "The following response answers a multiple-choice question (A, B, C, or D), but includes additional reasoning. "
                    "Please extract only the final answer choice (A, B, C, or D).",
                    response.text
                ]
            )
        return response.text
    else:
        contents = [
            question,
            f"{condition_modality}:",
            condition_data,
            "A:",
            choise_data['A'],
            "B:",
            choise_data['B'],
            "C:",
            choise_data['C'],
            "D:",
            choise_data['D'],
            'Please provide a detailed reasoning and then give the letter of the correct answer (A, B, C, or D) in a json format like this: {\"answer\": \"A\", \"reasoning\": \"...\"}.'
        ]
        
        print(f"Running model: {args.model}") 
        response = client.models.generate_content(
            model=args.model,
            # model="gemini-2.5-flash",
            # model=args.model, # model="gemini-2.0-flash",  # or "gemini-2.5-flash"
            contents=contents,
            config={
                    "response_mime_type": "application/json",
                    "response_schema": AnswerSchema,
                }
        )

        try:
            output = response.text
            answer = response.parsed.answer.strip().upper()
            reasoning = response.parsed.reasoning.strip()
        
        except Exception as e:
            print(f"Response is not a valid answer: {response.text.strip()}. Re-running with reasoning extraction.")
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                # model="gemini-2.5-flash",
                # model=args.model, # model="gemini-2.0-flash",  # or "gemini-2.5-flash"
                contents = [
                    "For the following response, extract the final answer choice (A, B, C, or D) and the reasoning into a json format like this: {\"answer\": \"A\", \"reasoning\": \"...\"}.",
                    response.text
                ]
            )  
            output = eval(response.text.strip())
            answer = output.get('answer', '').strip().upper()
            reasoning = output.get('reasoning', '').strip()

        return {
            'answer': answer,
            'reasoning': reasoning
        }

def run_genimi(questions, index, args):
    # instance = get_question_audio_vision_text(questions, index)
    instance = get_question(questions, index)
    
    try:
        response = _run_genimi(instance, args)
        return response
    except Exception as e:
        print(f"Error processing question {index}: {e}")
        return None
    
def run_all_genimi(task_name, questions, args, sample = 100, save_dir = None):
    correct_count = 0
    all_count = 0
    save_result = {}
    save_result['task_name'] = task_name
    save_result['score'] = 0
    save_result['correct_count'] = 0
    save_result['all_count'] = 0
    
    save_result['results'] = {}
    
    # task_name = task_name.replace('/', '_')
    task_name2 = task_name.split('_')
    modality_name = '_'.join(task_name2[-2:])
    task_name2 = '_'.join(task_name2[:-2])
    
    # load hard case
    hard_case_path = f"/home/xwang378/scratch/2025/AudioBench/benchmark/results/gemini-2.5-pro/hard_case.json"
    with open(hard_case_path, "r") as f:
        hard_case = json.load(f)
    hard_case_ids = [int(id) for id in hard_case[task_name2].keys()]
    
    if sample > len(questions):
        print(f"Sample is greater than the number of questions, setting sample to {len(questions)}")
        sample = len(questions)
    if sample == -1:
        sample = len(questions)

    
    for i in tqdm(range(sample)):
        if i not in hard_case_ids[:20]:
            continue
        response = run_genimi(questions, i, args)
        original_response = response
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
                is_correct = False
            print(f"Question {i}: {response.strip()} (Correct: {questions[i]['correct_answer']}); Current Score: {correct_count}/{all_count} = {correct_count / all_count * 100:.5f}%")
    
    
        save_result['results'][i] = {
            "question": questions[i]['question'],
            "response": response.strip() if response else None,
            'reasoning': reasoning,
            'original_response': original_response,
            "correct_answer": questions[i]['correct_answer'],
            "index": i,
            "is_correct": is_correct if response else False
        }
        
    save_result['score'] = correct_count / all_count * 100 if all_count > 0 else 0
    save_result['correct_count'] = correct_count
    save_result['all_count'] = all_count
    
    if args.reason:
        save_path = f"{save_dir}/{task_name.replace('/', '_')}_reason.json"
    else:
        save_path = f"{save_dir}/{task_name.replace('/', '_')}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(save_result, f, indent=4)
        
    print(f"Results saved to {save_path}")

def main(args):

    audiobench = AudioBench(root_dir=args.root_dir)

    task_name = args.task_name
    
    if 'audio_vision_text' in task_name:
        task_path_at = audiobench(task_name.replace('audio_vision_text', 'audio_text'))
        task_path_vt = audiobench(task_name.replace('audio_vision_text', 'vision_text'))
        questions = load_dual_questions(task_path_at, task_path_vt)
    else:
        task_path = audiobench(task_name)
        questions = load_questions(task_path)

    run_all_genimi(task_name, questions, args, sample=args.sample, save_dir=args.save_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', help='Root directory of AudioBench tasks', default='/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/')
    parser.add_argument('--task_name', help='Name of the task', default='perception/vggss_audio_vision')
    parser.add_argument('--sample', type=int, help='Number of samples to run', default=1)
    parser.add_argument('--save_dir', help='Directory to save results', default='/home/xwang378/scratch/2025/AudioBench/benchmark/results/')
    parser.add_argument('--model', help='Model to use for generation', default='gemini-2.0-flash')
    parser.add_argument('--reason', type=bool, default=False, help='Whether to run the reason script')
    args = parser.parse_args()
    main(args)
    


    
    