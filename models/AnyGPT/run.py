import os
import sys

sys.path.append("./")
sys.path.append("./anygpt/src")
sys.path.append("/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/")
sys.path.append("/home/xwang378/scratch/2025/AudioBench/models/AnyGPT/anygpt/src")
sys.path.append("/home/xwang378/scratch/2025/AudioBench/models/AnyGPT/anygpt/src/infer")

import re
import json
import random
import torch
import torchaudio
from einops import rearrange
import logging
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, EncodecModel, AutoProcessor
from seed2.seed_llama_tokenizer import ImageTokenizer
from speechtokenizer import SpeechTokenizer
from m_utils.anything2token import *
from m_utils.read_modality import encode_music_by_path
from m_utils.conversation import get_conv_template
from m_utils.prompter import *
from voice_clone import load_soundstorm, semantic2acoustic
from infer.pre_post_process import extract_content_between_final_tags
from PIL import Image
from datetime import datetime

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
client = genai.Client(api_key=API['Google_API_Key'])

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# ==================== 1. AnyGPT Chat Inference Model ====================

class AnyGPTChatInference:
    def __init__(
        self, 
        model_name_or_path: str="visual_inter_speech_golden_fs/checkpoint-30000",
        image_tokenizer_path: str="models/seed-tokenizer-2/seed_quantizer.pt",
        output_dir="infer_output/test",
        speech_tokenizer_path:str="models/speechtokenizer/ckpt.dev",
        speech_tokenizer_config:str="models/speechtokenizer/config.json",
        soundstorm_path:str="models/soundstorm/mls_1.pt"
    ):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.prompter = Prompter()

        # image_tokenizer
        print("loading image tokenzier")
        if image_tokenizer_path:
            self.image_tokenizer = ImageTokenizer(model_path=image_tokenizer_path, load_diffusion=True,
                                                  diffusion_model_path="stabilityai/stable-diffusion-2-1-unclip", device=self.device, image_size=224)
        print("loading speech tokenzier")
        self.speech_tokenizer = SpeechTokenizer.load_from_checkpoint(speech_tokenizer_config, speech_tokenizer_path)     
        self.speech_tokenizer.eval()
        self.speech_tokenizer.to(device=self.device)
        self.soundstorm = load_soundstorm(soundstorm_path)
        self.soundstorm.eval()
        self.soundstorm.to(device=self.device)
        print("loading music tokenizer")
        self.music_tokenizer = EncodecModel.from_pretrained("facebook/encodec_32khz")
        self.music_tokenizer.eval()
        self.music_tokenizer.to(device=self.device)
        self.music_processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")
        self.music_sample_rate = 32000
        self.music_segment_duration = 5
        print("loading audio tokenizer")
        self.audio_tokenizer = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.audio_tokenizer.eval()
        self.audio_tokenizer.to(device=self.device)
        self.audio_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.audio_sample_rate = 24000
        self.audio_segment_duration = 5
        
        # model
        print("loading llm")
        self.model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            )
        self.model.half()  
        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
        #tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left" 

        #generation
        self.output_dir = output_dir


    def encode_image(self, image_path=None, image_pil=None, image_torch=None):
        assert (image_path is None) + (image_pil is None) + (image_torch is None) == 2

        if image_path is not None:
            image_pil = Image.open(image_path).convert('RGB')

        if image_pil is not None:
            image_torch = self.image_tokenizer.processor(image_pil)
            image_torch = image_torch.to(self.device)
        return self.image_tokenizer.encode(image_torch)
    
    def encode_speech(self, audio_path):
        wav, sr = torchaudio.load(audio_path)
        # monophonic checking
        if wav.shape[0] > 1:
            wav = wav[:1, ]
        if sr != self.speech_tokenizer.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.speech_tokenizer.sample_rate)
        wav = wav.unsqueeze(0).to(self.device)
        # Extract discrete codes from SpeechTokenizer
        with torch.no_grad():
            codes = self.speech_tokenizer.encode(wav) # codes: (n_q, B, T)
        return codes[0, 0, :]
    
    def encode_media(self, media_path, modality, audio_as_speech=None, task_name=""):
        """Encode media based on modality using AnyGPT tokenizers"""
        
        # Handle audio modality mapping
        if modality.lower() == "audio":
            if audio_as_speech is None:
                # Auto-decide based on task name
                audio_as_speech = self.decide_audio_is_speech(task_name)
                print(f"Auto-detected audio_as_speech={audio_as_speech} for task: {task_name}")
            
            if audio_as_speech:
                modality = "Speech"
                print(f"Mapping audio to speech for encoding")
            else:
                print(f"Using audio encoding")
        
        if modality.lower() == "text":
            return media_path
        elif modality.lower() == "image":
            tokens = self.encode_image(image_path=media_path)[0]
        elif modality.lower() == "speech":
            print(f"Encoding speech: {media_path}")
            tokens = self.encode_speech(media_path)
        elif modality.lower() == "music":
            tokens = encode_music_by_path(
                media_path, self.music_sample_rate, self.music_tokenizer, 
                self.music_processor, self.device, 
                segment_duration=self.music_segment_duration, 
                one_channel=True, start_from_begin=True
            )
            tokens = tokens[0][0]
        elif modality.lower() == "audio":
            tokens = encode_music_by_path(
                media_path, self.audio_sample_rate, self.audio_tokenizer, 
                self.audio_processor, self.device, 
                segment_duration=self.audio_segment_duration, 
                one_channel=True, start_from_begin=True
            )
            tokens = tokens[0][0]
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        
        return modality_tokens_to_string(tokens=tokens, modality=modality.lower())

    def decide_audio_is_speech(self, task_name):
        """Decide if audio should be treated as speech based on task name"""
        speech_tasks = [
            'speech', 'speaker', 'voice', 'asr', 'recognition', 
            'tts', 'synthesis', 'spoken', 'utterance', 'phoneme',
            'word', 'sentence', 'conversation', 'dialogue', 
            'accent', 'pronunciation', 'linguistic', 'emotion'
        ]
        
        task_lower = task_name.lower()
        return any(keyword in task_lower for keyword in speech_tasks)

    def answer_audiobench_question(self, question_data, audio_type=None):
        """Answer a single AudioBench question using conversation format"""
        try:
            conversation = get_conv_template('MMGPT')
            
            # Extract question components
            question = question_data['question']
            condition_modality = question_data['condition_modality']
            condition_path = question_data['condition_path']
            choices_type = question_data['choices_type']
            choices_paths = question_data['choices_paths']
            
            # Encode condition and choices
            condition_tokens = self.encode_media(condition_path, condition_modality, audio_type)
            choice_tokens = []
            for choice_path in choices_paths:
                choice_token = self.encode_media(choice_path, choices_type, audio_type)
                choice_tokens.append(choice_token)
            
            # Create instruction for multiple choice question
            if condition_modality != 'Text':
                instruction = {condition_tokens}
                instruction = f" {question}\n"
            else:
                instruction = f"{question}\n"
                instruction += f"{condition_modality}: {condition_tokens}\n"
            
            # instruction += "Options:\n"
            for i, choice_token in enumerate(choice_tokens):
                choice_letter = chr(ord('A') + i)
                instruction += f"{choice_letter}: {choice_token}."
            instruction += " Please provide the letter (A, B, C, or D) of the correct choice."
            
            # Use the prompter to generate instruction prompt
            prompt_seq = self.prompter.generate_insturction_prompt(
                task="customized", 
                instruction=instruction, 
                image_list=[], 
                speech_list=[], 
                music_list=[]
            ).strip()
                        
            conversation.append_message(conversation.roles[0], prompt_seq)
            prompt = conversation.get_prompt()
            print(prompt)
            # Generate response
            
            input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True).input_ids
            input_ids = input_ids.to(self.device)
            
            # Use text generation config
            config_dict = json.load(open('models/AnyGPT/config/generate_config.json', 'r'))
            generation_config = GenerationConfig(**config_dict)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            
            generated_ids = generated_ids.sequences
            response = self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0]
            # import ipdb; ipdb.set_trace()
            # Extract answer using the same method as chat inference
            try:
                # response = response.split('<sosp>')[0].split('<soim>')[0]
                try:
                    response = extract_content_between_final_tags(response, tag1=f"{chatbot_name}", tag2="<eom>").strip()
                except:
                    try:
                        response = extract_content_between_final_tags(response, tag1=f"{chatbot_name}", tag2="<eos>").strip()
                    except:
                        response = response.split('<sosp>')[0].split('<soim>')[0]

            except Exception as e:
                print(f"Error in extract_answer: {e}")

            return response
        except Exception as e:
            print(f"Error in answer_audiobench_question: {e}")
            return "Error"

    def extract_answer(self, response):
        """Extract answer choice from model response"""
        # Look for pattern like "A", "B", "C", "D" 
        # if not, use gemini to extract the answer
        match = response.strip().upper()
        return match
        
def load_model_and_processor():
    """Load AnyGPT Chat model and create processor object for compatibility"""
    # Use default paths - these should be updated based on your setup
    model_path = "models/AnyGPT/models/anygpt/chat"  # Update this path
    image_tokenizer_path = "models/AnyGPT/models/seed-tokenizer-2/seed_quantizer.pt"
    speech_tokenizer_path = "models/AnyGPT/models/speechtokenizer/ckpt.dev"
    speech_tokenizer_config = "models/AnyGPT/models/speechtokenizer/config.json"
    soundstorm_path = "models/AnyGPT/models/soundstorm/speechtokenizer_soundstorm_mls.pt"
    
    anygpt_chat = AnyGPTChatInference(
        model_name_or_path=model_path,
        image_tokenizer_path=image_tokenizer_path,
        speech_tokenizer_path=speech_tokenizer_path,
        speech_tokenizer_config=speech_tokenizer_config,
        soundstorm_path=soundstorm_path
    )
    
    # Create a processor object for compatibility
    processor = {
        'anygpt_chat': anygpt_chat
    }
    
    return anygpt_chat, processor

def reevaluate_model(answer):
    print(f"Re-evaluating model response: {answer}")
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents = [
            "The following response answers a multiple-choice question (A, B, C, or D), but includes additional words. ",
            f"Response: {answer}",
            "Please extract the best matching answer choice (A, B, C, or D).",
            
        ]
    ) 
    answer = response.text.strip().upper()
    return answer

def run_model(model, processor, conversation, audio_type=None):
    """Run AnyGPT Chat model with conversation input"""
    try:
        # Extract question data from conversation format
        user_content = conversation[1]['content']
        
        # Parse the conversation to extract question, condition, and choices
        question = user_content[0]['text']
        
        condition_modality = user_content[2]['type'].capitalize()
        condition_path = user_content[2][condition_modality.lower()]

        choices_modality = user_content[4]['type'].capitalize()
        choices_paths = []
        for i in [4, 6, 8, 10]:
            choices_paths.append(user_content[i][choices_modality.lower()])
        
        if condition_modality == 'Audio':
            condition_modality = audio_type
        if choices_modality == 'Audio':
            choices_modality = audio_type

        instruction = user_content[-1]['text']
            
        question_data = {
            'question': question,
            'condition_modality': condition_modality,
            'condition_path': condition_path,
            'choices_type': choices_modality,
            'choices_paths': choices_paths
        }
        
        # Use AnyGPT Chat to answer the question
        answer = model.answer_audiobench_question(question_data, audio_type)
        
        return answer, None  # Return None for audio to match interface
        
    except Exception as e:
        print(f"Error in run_model: {e}")
        return "Error", None

# ==================== 2. Generate conversation for compatibility ====================

def decide_audio_type(task_name):
    """Decide audio type based on task name"""
    speech_tasks = ['speech', 'emotion', 'translation', 'recognition']
    music_tasks = ['music', 'singer']
    
    task_lower = task_name.lower()
    
    if any(keyword in task_lower for keyword in speech_tasks):
        return 'Speech'
    elif any(keyword in task_lower for keyword in music_tasks):
        return 'Music'
    else:
        return 'Audio'

def process_media_info(media_type, media_path, audio_type='Audio'):
    if media_type == 'Video':
        return {
            "type": "video",
            "video": media_path,
            "fps": 12,
            "max_frames": 12*5,
            "max_pixels": 512 * 512,
        }
    elif media_type == 'Image':
        return {
            "type": "image",
            "image": media_path,
            "max_pixels": 512 * 512,
        }
    elif media_type == 'Audio':
        return {
            "type": audio_type,
            audio_type: media_path,
        }
    else:
        return {
            "type": media_type,
            media_type: media_path,
        }

def generate_conversation(question, reason=False, audio_type='Audio'):
    condition_modality = question['condition_modality'].lower()
    choises_type = question['choices_type'].lower()
    
    if not args.reason:
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are AnyGPT, a multimodal AI assistant capable of understanding and generating text, images, audio, speech, and music."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question['question']},
                    {"type": "text", "text": f"{question['condition_modality']}:"},
                    process_media_info(condition_modality, question['condition_path'], audio_type),
                    {"type": "text", "text": "A:"},
                    process_media_info(choises_type, question['choices_paths'][0], audio_type),
                    {"type": "text", "text": "B:"},
                    process_media_info(choises_type, question['choices_paths'][1], audio_type),
                    {"type": "text", "text": "C:"},
                    process_media_info(choises_type, question['choices_paths'][2], audio_type),
                    {"type": "text", "text": "D:"},
                    process_media_info(choises_type, question['choices_paths'][3], audio_type),
                    {"type": "text", "text": "Give the letter of the correct answer (A, B, C, or D)."}
                ]
            },
        ]
    else:
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are AnyGPT, a multimodal AI assistant capable of understanding and generating text, images, audio, speech, and music."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question['question']},
                    {"type": "text", "text": f"{question['condition_modality']}:"},
                    process_media_info(condition_modality, question['condition_path'], audio_type),
                    {"type": "text", "text": "A:"},
                    process_media_info(choises_type, question['choices_paths'][0], audio_type),
                    {"type": "text", "text": "B:"},
                    process_media_info(choises_type, question['choices_paths'][1], audio_type),
                    {"type": "text", "text": "C:"},
                    process_media_info(choises_type, question['choices_paths'][2], audio_type),
                    {"type": "text", "text": "D:"},
                    process_media_info(choises_type, question['choices_paths'][3], audio_type),
                    {"type": "text", "text":'Please provide a detailed reasoning and then give the letter of the correct answer (A, B, C, or D) in a json format like this: {\"answer\": \"A\", \"reasoning\": \"...\"}.'}
                ]
            },
        ]

# ==================== 3. Main execution ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', help='Root directory of AudioBench tasks', default='/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/')
    parser.add_argument('--task_name', help='Name of the task', default='perception/vggss_audio_vision')
    parser.add_argument('--sample', type=int, help='Number of samples to run', default=1)
    parser.add_argument('--save_dir', help='Directory to save results', default='/home/xwang378/scratch/2025/AudioBench/benchmark/results/')
    parser.add_argument('--reason', type=bool, default=False, help='Whether to run the reason script')
    parser.add_argument('--audio_as_speech', type=bool, default=None, 
                       help='Whether to treat audio as speech. If None, auto-decide based on task name')
    args = parser.parse_args()
    
    if args.reason:
        print(f"Running with reason")
    
    if args.audio_as_speech is not None:
        print(f"Audio will be treated as {'speech' if args.audio_as_speech else 'audio'} (manual setting)")
    else:
        print(f"Audio modality will be auto-detected based on task name")
    
    audiobench = AudioBench(root_dir=args.root_dir)

    task_name = args.task_name
    task_path = audiobench(task_name)
    questions = load_questions(task_path)
    
    audio_type = decide_audio_type(task_name)
    print(f"Use Audio Type: {audio_type} for task {task_name}")
    
    # Load model and processor once
    model, processor = load_model_and_processor()

    correct_count = 0
    all_count = 0
    format_error_count = 0
    run_error_count = 0
    save_result = {}
    save_result['task_name'] = task_name
    save_result['score'] = 0
    save_result['correct_count'] = 0
    save_result['all_count'] = 0
    
    save_result['results'] = {}
    
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
    
    # if os.path.exists(os.path.join(args.save_dir, f"{task_name.replace('/', '_')}.json")):
    #     print(f"Results already exist for {task_name}, loading existing results...")
    #     exist_result = json.load(open(os.path.join(args.save_dir, f"{task_name.replace('/', '_')}.json")))
    #     save_result['results'] = exist_result.get('results', {})

    
    for i in tqdm(sample_range):
        instance = get_question(questions, i)
        correct_answers = instance['correct_answer']
        
        previous_result = save_result['results'].get(str(i), None)
        if previous_result is None or previous_result['original_text'].startswith('Error'):
            try:
                conversation = generate_conversation(instance, reason=args.reason, audio_type=audio_type)
                text, audio = run_model(model, processor, conversation, 
                                      audio_type=audio_type)
                if args.reason:
                    try:
                        output = eval(text)
                        reasoning = output.get('reasoning', '')
                        text = output.get('answer', '')
                    except Exception as e:
                        print(f"Error parsing reasoning: {e}")
                        reasoning = ''
                        text = f'Error: {e}'
            except Exception as e:
                print(f"Error generating conversation: {e}")
                text = f'Error: {e}'
                audio = None
        else:
            text = previous_result['original_text']
            correct_answers = previous_result['correct_answer']
        
        original_text = text
        if text.strip().upper() not in ['A', 'B', 'C', 'D']:
            text = reevaluate_model(text)

        print(f"Question: {i}, GT: {correct_answers}, Answer: {text}")
        
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
            "original_text": original_text,
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
    
    if not args.reason:
        save_path = os.path.join(args.save_dir, f"{task_name.replace('/', '_')}.json")
    else:
        save_path = os.path.join(args.save_dir, f"{task_name.replace('/', '_')}_reason.json")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(save_result, f, indent=4)
    print(f"Results saved to {save_path}")