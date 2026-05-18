import sys
sys.path.append("/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/")
import os
import json

from audioBench import AudioBench
import argparse
from tqdm import tqdm
import random

print("Using random model for answer generation")

def load_questions(path):
    with open(path, 'r') as f:
        questions = json.load(f)
    return questions

def load_dual_questions(path_at, path_vt):
    questions_at = load_questions(path_at)
    questions_vt = load_questions(path_vt)
    questions = []
    for question_at, question_vt in zip(questions_at, questions_vt):
        # Create a simple combined question for random model
        combined_question = f"Given both audio and visual input, choose the correct option."
        question = {
            "question": combined_question,
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
    
def _run_random(instance, args):
    # Generate random answer from A, B, C, D
    random_answer = random.choice(['A', 'B', 'C', 'D'])
    
    if not args.reason:
        # print(f"Random model generated answer: {random_answer}")
        return random_answer
    else:
        # Generate random reasoning for the random answer
        random_reasonings = [
            "Based on random selection, this seems like the most likely choice.",
            "Random analysis suggests this option is correct.",
            "Using random decision-making process, this appears to be the answer.",
            "Random evaluation indicates this is the best option.",
            "Through random assessment, this choice stands out."
        ]
        
        random_reasoning = random.choice(random_reasonings)
        print(f"Random model generated answer: {random_answer} with reasoning")
        
        return {
            'answer': random_answer,
            'reasoning': random_reasoning
        }

def run_random_model(questions, index, args):
    # instance = get_question_audio_vision_text(questions, index)
    instance = get_question(questions, index)
    
    try:
        response = _run_random(instance, args)
        return response
    except Exception as e:
        print(f"Error processing question {index}: {e}")
        return None
    
def run_all_random(task_name, questions, args, sample = 100, save_dir = None):
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

    
    if sample > len(questions):
        print(f"Sample is greater than the number of questions, setting sample to {len(questions)}")
        sample = len(questions)
    if sample == -1:
        sample = len(questions)

    
    for i in tqdm(range(sample)):

        response = run_random_model(questions, i, args)
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
            # print(f"Question {i}: {response.strip()} (Correct: {questions[i]['correct_answer']}); Current Score: {correct_count}/{all_count} = {correct_count / all_count * 100:.5f}%")
    
    
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

    run_all_random(task_name, questions, args, sample=args.sample, save_dir=args.save_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', help='Root directory of AudioBench tasks', default='/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/')
    parser.add_argument('--task_name', help='Name of the task', default='perception/vggss_audio_vision')
    parser.add_argument('--sample', type=int, help='Number of samples to run', default=1)
    parser.add_argument('--save_dir', help='Directory to save results', default='/home/xwang378/scratch/2025/AudioBench/benchmark/results/')
    parser.add_argument('--model', help='Model to use for generation', default='random')
    parser.add_argument('--reason', type=bool, default=False, help='Whether to run the reason script')
    args = parser.parse_args()
    main(args)
    


    