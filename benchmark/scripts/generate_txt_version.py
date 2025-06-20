import json


vision_audio = '/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/vggss/vggss_audio_bench_questions_vision_audio.json'
audio_vision = '/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/vggss/vggss_audio_bench_questions_audio_vision.json'

vision_text = '/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/vggss/vggss_audio_bench_questions_vision_text.json'
audio_text = '/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/01_perception/vggss/vggss_audio_bench_questions_audio_text.json'

with open(vision_audio, 'r') as f:
    vision_audio_data = json.load(f)
    
with open(audio_vision, 'r') as f:
    audio_vision_data = json.load(f)
    
vggss_anno = '/home/xwang378/scratch/2025/AudioBench/benchmark/Data/vggss.json'
with open(vggss_anno, 'r') as f:
    vggss_data = json.load(f)
    
id2class = {}
for item in vggss_data:
    id2class[item['file']] = item['class']   

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
        index = index.split('/')[-2].split('_frames')[0]
        
        text = id2class[index]
        
        item['options'][option]['modality'] = 'Text'
        item['options'][option]['input'] = text.capitalize()
        
with open(audio_text, 'w') as f:
    json.dump(audio_vision_data, f, indent=4)
         
            