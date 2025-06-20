import json
import os

class AudioBench():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        self.task_paths = {
            'perception': {
                'vggss_audio_vision': os.path.join(root_dir, '01_perception', 'vggss', 'vggss_audio_bench_questions_audio_vision.json'),
                'vggss_vision_audio': os.path.join(root_dir, '01_perception', 'vggss', 'vggss_audio_bench_questions_vision_audio.json'),
                'vggss_audio_text': os.path.join(root_dir, '01_perception', 'vggss', 'vggss_audio_bench_questions_audio_text.json'),
                'vggss_vision_text': os.path.join(root_dir, '01_perception', 'vggss', 'vggss_audio_bench_questions_vision_text.json')
            }
        }
        
    def __call__(self, task_name):
        name_1, name_2 = task_name.split('/')
        
        return self.task_paths[name_1][name_2]
        
        
    def load_task_data(self, task_name):
        if task_name not in self.task_paths:
            raise ValueError(f"Task {task_name} is not defined in AudioBench.")
        
        task_data = {}
        for subtask, path in self.task_paths[task_name].items():
            if os.path.exists(path):
                with open(path, 'r') as f:
                    task_data[subtask] = json.load(f)
            else:
                print(f"Warning: Path {path} does not exist.")
        
        
        return task_data
    

