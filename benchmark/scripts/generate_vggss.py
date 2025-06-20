import os
import re
import random
import torch
import soundfile as sf
import json

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

print("Available GPUs:", torch.cuda.device_count())



# ==================== 1. Main function: Load model and answer one question ====================

def load_model_and_processor():
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    return model, processor


def generate_conversation_vision( audio_choices, audio_paths, correct_answer):
    
    image_frames_folders = [audio_sample.replace('.wav', '_frames') for audio_sample in audio_paths]
    image_paths = [os.path.join(folder, os.listdir(folder)[0]) for folder in image_frames_folders]
    
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Which image most likely belongs to the object that make this sound? Answer the question with A, B, C, or D"},
                {"type": "audio", "audio": correct_audio_sample},
                {"type": "text", "text": "A:"},
                {"type": "image", "image": image_paths[0]},
                {"type": "text", "text": "B:"},
                {"type": "image", "image": image_paths[1]},
                {"type": "text", "text": "C:"},
                {"type": "image", "image": image_paths[2]},
                {"type": "text", "text": "D:"},
                {"type": "image", "image": image_paths[3]},
            ],
        },
    ]
    
    question = {
        "question": "Which image most likely belongs to the object that make this sound? Answer the question with A, B, C, or D",
        "conditions": {
            "modality": "Audio",
            "input": correct_audio_sample,
            },
        "options": {
            "A": {
                "modality": "Image",
                "input": image_paths[0],
            },
            "B": {
                "modality": "Image",
                "input": image_paths[1],
            },
            "C": {
                "modality": "Image",
                "input": image_paths[2],
            },
            "D": {
                "modality": "Image",
                "input": image_paths[3],
            }
        },
        "correct_answer": correct_answer
    }
    
    return conversation, question


def generate_conversation_audio(audio_choices, audio_paths, correct_answer):
    
    correct_audio_sample = audio_paths[ord(correct_answer) - ord('A')]
    
    image_frames_folder = correct_audio_sample.replace('.wav', '_frames')
    image_path = os.path.join(image_frames_folder, os.listdir(image_frames_folder)[0])  # Use the first frame as the image
    
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Which sound is most likely made by the object from this image? Answer the question with A, B, C, or D"},
                {"type": "image", "image": image_path},
                {"type": "text", "text": "A."},
                {"type": "audio", "audio": audio_paths[0]},
                {"type": "text", "text": "B."},
                {"type": "audio", "audio": audio_paths[1]},
                {"type": "text", "text": "C."},
                {"type": "audio", "audio": audio_paths[2]},
                {"type": "text", "text": "D."},
                {"type": "audio", "audio": audio_paths[3]},
            ],
        },
    ]
    
    question = {
        "question": "Which sound is most likely made by the object from this image? Answer the question with A, B, C, or D",
        "conditions": {
            "modality": "Image",
            "input": image_path,
            },
        "options": {
            "A": {
                "modality": "Audio",
                "input": audio_paths[0],
            },
            "B": {
                "modality": "Audio",
                "input": audio_paths[1],
            },
            "C": {
                "modality": "Audio",
                "input": audio_paths[2],
            },
            "D": {
                "modality": "Audio",
                "input": audio_paths[3],
            }
        },
        "correct_answer": correct_answer
    }
    
    return conversation, question

def run_model(model, processor, conversation, use_audio_in_video=False):
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    text_ids, audio = model.generate(**inputs, use_audio_in_video=use_audio_in_video)
    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    match = re.search(r'assistant\s*\n([A-D])\.', text[0])
    
    if match:
        answer = match.group(1)
    else:
        answer = "No answer found"
    return answer, audio



# ==================== 3. Example usage loop ====================

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    # Load model and processor once
    model, processor = load_model_and_processor()

    # Set fixed audio choices for all questions
    all_audio_choices = os.listdir("/home/xwang378/scratch/Data/vggss_audio_bench/")
    all_audio_choices = [audio_name for audio_name in all_audio_choices if audio_name.endswith('.wav')]
    

    # Directory of frames
    valid_all_audio_choices = []
    for audio_name in all_audio_choices:
        frames_folder = audio_name.replace('.wav', '_frames')
        frames_path = os.path.join("/home/xwang378/scratch/Data/vggss_audio_bench/", frames_folder)
        if not os.path.exists(frames_path):
            print(f"Frames folder does not exist for {audio_name}, skipping.")
            continue
        if len(os.listdir(frames_path)) == 0:
            print(f"No frames found in {frames_folder}, skipping.")
            continue
        valid_all_audio_choices.append(audio_name)
    all_audio_choices = valid_all_audio_choices
    
    print(f"Total valid audio choices: {len(all_audio_choices)}")
    
    
    
    correct_vision = 0
    correct_audio = 0
    all_sample = 0
    
    audio_questions = []
    vision_questions = []
    for i in range(1000):
        audio_choices = random.sample(all_audio_choices, 4)
        audio_paths = [os.path.join("/home/xwang378/scratch/Data/vggss_audio_bench/", audio_name) for audio_name in audio_choices]

        correct_answers = random.choice(['A', 'B', 'C', 'D'])
        
        conversation_vision, question_vision = \
            generate_conversation_vision(audio_choices, audio_paths, correct_answers)
        text_vision, audio = run_model(model, processor, conversation_vision)
        
        conversation_audio, question_audio = \
            generate_conversation_audio(audio_choices, audio_paths, correct_answers)
        text_audio, audio = run_model(model, processor, conversation_audio)

        print(f"GT: {correct_answers}, Answer Vision: {text_vision}, Answer Audio: {text_audio}")
        
        audio_questions.append(question_audio)
        vision_questions.append(question_vision)
        
        if text_vision == correct_answers:
            correct_vision += 1
        
        if text_audio == correct_answers:
            correct_audio += 1
            
        all_sample += 1
        
        print(f"Vision Accuracy: {correct_vision / all_sample * 100:.2f}. {correct_vision}/{all_sample}")
        print(f"Audio Accuracy: {correct_audio / all_sample * 100:.2f}. {correct_audio}/{all_sample}")
        
                
        # save log/
        with open("vggss_audio_bench_log.txt", "a") as f:
            f.write(f"GT: {correct_answers}, Answer Vision: {text_vision}, Answer Audio: {text_audio}, "
                    f"Vision Accuracy: {correct_vision / all_sample * 100:.2f}. {correct_vision}/{all_sample}, "
                    f"Audio Accuracy: {correct_audio / all_sample * 100:.2f}. {correct_audio}/{all_sample}\n")
    # Save all questions to a file,  dump json
    
    with open("vggss_audio_bench_questions_audio.json", "w") as f:
        json.dump(audio_questions, f, indent=4)
    with open("vggss_audio_bench_questions_vision.json", "w") as f:
        json.dump(vision_questions, f, indent=4)
    