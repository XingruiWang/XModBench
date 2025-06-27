import json
import os


# DATRASET_NAME = 'solos'
# DATRASET_NAME = 'URMP'
DATRASET_NAME = 'landscapes'

vision_audio = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/{DATRASET_NAME}/{DATRASET_NAME}_audio_bench_questions_vision_audio.json'
audio_vision = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/{DATRASET_NAME}/{DATRASET_NAME}_audio_bench_questions_audio_vision.json'

vision_text = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/{DATRASET_NAME}/{DATRASET_NAME}_audio_bench_questions_vision_text.json'
audio_text = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/{DATRASET_NAME}/{DATRASET_NAME}_audio_bench_questions_audio_text.json'

text_vision = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/{DATRASET_NAME}/{DATRASET_NAME}_audio_bench_questions_text_vision.json'
text_audio = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/{DATRASET_NAME}/{DATRASET_NAME}_audio_bench_questions_text_audio.json'


with open(vision_audio, 'r') as f:
    vision_audio_data = json.load(f)
    
with open(audio_vision, 'r') as f:
    audio_vision_data = json.load(f)

    

id2class = {}
# data_root = f'/home/xwang378/scratch/2025/AudioBench/benchmark/Data/{DATRASET_NAME}_processed'
data_root = '/home/xwang378/scratch/2025/AudioBench/benchmark/Data/landscape_audiobench/test_processed'

URMP_id2class = {
    'vn': 'Violin',
    'vc': 'Cello',
    'va': 'Viola',
    'fl': 'Flute',
    'cl': 'Clarinet',
    'tpt': 'Trumpet',
    'sax': 'Saxophone',
    'tbn': 'Trombone',
    'tba': 'Tuba',
    'ob': 'Oboe',
    'hn': 'French Horn',
    'db': 'Double Bass',
    'bn': 'Bassoon'}
    
def make_description(objects_name):
    if len(objects_name) == 1:
        return f"A {objects_name[0].capitalize()}."
    elif len(objects_name) == 2:
        return f"A {objects_name[0].capitalize()} and a {objects_name[1].capitalize()}."
    elif len(objects_name) >= 3:
        return f"A {', '.join(objects_name[:-1])}, and a {objects_name[-1].capitalize()}."
    else:
        raise ValueError("Invalid number of objects in the description.")
        
for class_name in os.listdir(data_root):
    for item in os.listdir(os.path.join(data_root, class_name)):
        if item.endswith('.wav'):
            index = item.split('.')[0]
            if DATRASET_NAME == 'solos':
                id2class[index] = class_name
            elif DATRASET_NAME == 'URMP':
                objects_name = class_name.split('_')[2:]
                
                description = make_description([URMP_id2class[obj] for obj in objects_name])
                
                id2class[index] = description
            elif DATRASET_NAME == 'landscapes':
                objects_name = class_name.split('_')
                
                description = ' '.join(objects_name).capitalize()
                
                id2class[index] = description

vision_text_data = []

# Process vision to text
for item in vision_audio_data:
    # item['question'] = "Which phrase is most likely to describe the objects in this image? Answer the question with A, B, C, or D"
    # item['question'] = "Which phrase is most likely to describe this instrument composition in this image? Answer the question with A, B, C, or D"
    item['question'] = "Which phrase is most likely to describe this scene in this image? Answer the question with A, B, C, or D"
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
    # item['question'] = "Which phrase is most likely to describe the sound in this audio? Answer the question with A, B, C, or D."
    # item['question'] = "Which phrase is most likely to describe this instrument composition in this audio? Answer the question with A, B, C, or D."
    item['question'] = "Which phrase is most likely to describe this scene in this audio? Answer the question with A, B, C, or D."
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
    # item['question'] = "Which audio is most likely to belong to this description of this instrument composition? Answer the question with A, B, C, or D."
    item['question'] = "Which audio is most likely to belong to this description of this scene? Answer the question with A, B, C, or D."
    item['conditions']['modality'] = 'Text'
    
    index = item['conditions']["input"] 
    index = index.split('/')[-1].split('.')[0]
    item['conditions']['input'] = id2class[index].capitalize()
    
        
with open(text_audio, 'w') as f:
    json.dump(vision_audio_data, f, indent=4)     

# Process audio to text
for item in audio_vision_data:
    # item['question'] = "Which image is most likely to belong to this description of this instrument composition? Answer the question with A, B, C, or D."
    item['question'] = "Which image is most likely to belong to this description of this scene? Answer the question with A, B, C, or D."

    item['conditions']['modality'] = 'Text'
    index = item['conditions']["input"] 
    index = index.split('/')[-1].split('.')[0]
    
    item['conditions']['input'] = id2class[index].capitalize()
        
with open(text_vision, 'w') as f:
    json.dump(audio_vision_data, f, indent=4)
         
       