import json
import os

vision_audio = '/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/solos/solos_audio_bench_questions_vision_audio.json'
audio_vision = '/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/solos/solos_audio_bench_questions_audio_vision.json'

vision_text = '/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/solos/solos_audio_bench_questions_vision_text.json'
audio_text = '/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/solos/solos_audio_bench_questions_audio_text.json'

text_vision = '/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/solos/solos_audio_bench_questions_text_vision.json'
text_audio = '/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/solos/solos_audio_bench_questions_text_audio.json'


with open(vision_audio, 'r') as f:
    vision_audio_data = json.load(f)
    
with open(audio_vision, 'r') as f:
    audio_vision_data = json.load(f)
    
id2class = {}
data_root = '/home/xwang378/scratch/2025/AudioBench/benchmark/Data/solos_processed'
for class_name in os.listdir(data_root):
    for item in os.listdir(os.path.join(data_root, class_name)):
        if item.endswith('.wav'):
            index = item.split('.')[0]
            id2class[index] = class_name

vision_text_data = []

# Process vision to text
for item in vision_audio_data:
    item['question'] = "Which phrase is most likely to describe the object in this image? Answer the question with A, B, C, or D"
    for option in item['options']:
        index = item['options'][option]["input"] 
        index = index.split('/')[-1].split('.')[0]
        
        text = id2class[index]
        
        item['options'][option]['modality'] = 'Text'
        item['options'][option]['input'] = text.capitalize()
        
with open(vision_text, 'w') as f:
    json.dump(vision_audio_data, f, indent=4)     

# Process audio to text
for item in audio_vision_data:
    item['question'] = "Which phrase is most likely to describe the sound in this audio? Answer the question with A, B, C, or D."
    for option in item['options']:
        index = item['options'][option]["input"] 
        index = index.split('/')[-1].split('.')[0]
        
        text = id2class[index]
        
        item['options'][option]['modality'] = 'Text'
        item['options'][option]['input'] = text.capitalize()
        
with open(audio_text, 'w') as f:
    json.dump(audio_vision_data, f, indent=4)
         
         
############### Process condision ###############
with open(vision_audio, 'r') as f:
    vision_audio_data = json.load(f)
    
with open(audio_vision, 'r') as f:
    audio_vision_data = json.load(f)
           

            
# Process vision to text
for item in vision_audio_data:
    item['question'] = "Which audio is most likely to belong to this phrase? Answer the question with A, B, C, or D."
    item['conditions']['modality'] = 'Text'
    
    index = item['conditions']["input"] 
    index = index.split('/')[-1].split('.')[0]
    item['conditions']['input'] = id2class[index].capitalize()
    
        
with open(text_audio, 'w') as f:
    json.dump(vision_audio_data, f, indent=4)     

# Process audio to text
for item in audio_vision_data:
    item['question'] = "Which image is most likely to belong to this phrase? Answer the question with A, B, C, or D."

    item['conditions']['modality'] = 'Text'
    index = item['conditions']["input"] 
    index = index.split('/')[-1].split('.')[0]
    
    item['conditions']['input'] = id2class[index].capitalize()
        
with open(text_vision, 'w') as f:
    json.dump(audio_vision_data, f, indent=4)
         
       