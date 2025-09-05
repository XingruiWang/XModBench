import json
import os

class AudioBench():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        self.task_paths = {
            'perception': {
                'general_audio_vision': os.path.join(root_dir, '01_perception', 'general_activities', 'general_activities_questions_audio_vision.json'),
                'general_vision_audio': os.path.join(root_dir, '01_perception', 'general_activities', 'general_activities_questions_vision_audio.json'),
                'general_audio_text': os.path.join(root_dir, '01_perception', 'general_activities', 'general_activities_questions_audio_text.json'),
                'general_text_audio': os.path.join(root_dir, '01_perception', 'general_activities', 'general_activities_questions_text_audio.json'),
                'general_vision_text': os.path.join(root_dir, '01_perception', 'general_activities', 'general_activities_questions_vision_text.json'),
                'general_text_vision': os.path.join(root_dir, '01_perception', 'general_activities', 'general_activities_questions_text_vision.json'),
                
                'finegrained_audio_vision': os.path.join(root_dir, '01_perception', 'finegrained', 'vggss_questions_audio_vision.json'),
                'finegrained_audio_text': os.path.join(root_dir, '01_perception', 'finegrained', 'vggss_questions_audio_text.json'),
                'finegrained_text_audio': os.path.join(root_dir, '01_perception', 'finegrained', 'vggss_questions_text_audio.json'),
                'finegrained_text_vision': os.path.join(root_dir, '01_perception', 'finegrained', 'vggss_questions_text_vision.json'),
                'finegrained_vision_audio': os.path.join(root_dir, '01_perception', 'finegrained', 'vggss_questions_vision_audio.json'),
                'finegrained_vision_text': os.path.join(root_dir, '01_perception', 'finegrained', 'vggss_questions_vision_text.json'),
                
                'instruments_audio_vision': os.path.join(root_dir, '01_perception', 'instruments', 'solos_audio_bench_questions_audio_vision.json'),
                'instruments_vision_audio': os.path.join(root_dir, '01_perception', 'instruments', 'solos_audio_bench_questions_vision_audio.json'),
                'instruments_vision_text': os.path.join(root_dir, '01_perception', 'instruments', 'solos_audio_bench_questions_vision_text.json'),
                'instruments_audio_text': os.path.join(root_dir, '01_perception', 'instruments', 'solos_audio_bench_questions_audio_text.json'),
                'instruments_text_audio': os.path.join(root_dir, '01_perception', 'instruments', 'solos_audio_bench_questions_text_audio.json'),
                'instruments_text_vision': os.path.join(root_dir, '01_perception', 'instruments', 'solos_audio_bench_questions_text_vision.json'),
                
                'instruments_comp_audio_vision': os.path.join(root_dir, '01_perception', 'instruments_comp', 'URMP_questions_audio_vision.json'),
                'instruments_comp_vision_audio': os.path.join(root_dir, '01_perception', 'instruments_comp', 'URMP_questions_vision_audio.json'),
                'instruments_comp_audio_text': os.path.join(root_dir, '01_perception', 'instruments_comp', 'URMP_questions_audio_text.json'),
                'instruments_comp_text_audio': os.path.join(root_dir, '01_perception', 'instruments_comp', 'URMP_questions_text_audio.json'),
                'instruments_comp_vision_text': os.path.join(root_dir, '01_perception', 'instruments_comp', 'URMP_questions_vision_text.json'),
                'instruments_comp_text_vision': os.path.join(root_dir, '01_perception', 'instruments_comp', 'URMP_questions_text_vision.json'),
                
                'natures_audio_vision': os.path.join(root_dir, '01_perception', 'natures', 'landscapes_audio_bench_questions_audio_vision.json'),
                'natures_vision_audio': os.path.join(root_dir, '01_perception', 'natures', 'landscapes_audio_bench_questions_vision_audio.json'),
                'natures_audio_text': os.path.join(root_dir, '01_perception', 'natures', 'landscapes_audio_bench_questions_audio_text.json'),
                'natures_text_audio': os.path.join(root_dir, '01_perception', 'natures', 'landscapes_audio_bench_questions_text_audio.json'),
                'natures_vision_text': os.path.join(root_dir, '01_perception', 'natures', 'landscapes_audio_bench_questions_vision_text.json'),
                'natures_text_vision': os.path.join(root_dir, '01_perception', 'natures', 'landscapes_audio_bench_questions_text_vision.json'),
            },
            
            'spatial': {
                'arrangements_audio_vision': os.path.join(root_dir, '02_spatial', 'arrangements', 'urmp_audio_bench_questions_audio_vision.json'),
                'arrangements_vision_audio': os.path.join(root_dir, '02_spatial', 'arrangements', 'urmp_audio_bench_questions_vision_audio.json'),
                'arrangements_audio_text': os.path.join(root_dir, '02_spatial', 'arrangements', 'urmp_audio_bench_questions_audio_text.json'),
                'arrangements_text_audio': os.path.join(root_dir, '02_spatial', 'arrangements', 'urmp_audio_bench_questions_text_audio.json'),
                'arrangements_vision_text': os.path.join(root_dir, '02_spatial', 'arrangements', 'urmp_audio_bench_questions_vision_text.json'),
                'arrangements_text_vision': os.path.join(root_dir, '02_spatial', 'arrangements', 'urmp_audio_bench_questions_text_vision.json'),

                '3D_movements_audio_vision': os.path.join(root_dir, '02_spatial', '3D_movements', 'urbansas_audio_bench_questions_audio_vision.json'),
                '3D_movements_vision_audio': os.path.join(root_dir, '02_spatial', '3D_movements', 'urbansas_audio_bench_questions_vision_audio.json'),
                '3D_movements_audio_text': os.path.join(root_dir, '02_spatial', '3D_movements', 'urbansas_audio_bench_questions_audio_text.json'),
                '3D_movements_text_audio': os.path.join(root_dir, '02_spatial', '3D_movements', 'urbansas_audio_bench_questions_text_audio.json'),
                '3D_movements_vision_text': os.path.join(root_dir, '02_spatial', '3D_movements', 'urbansas_audio_bench_questions_vision_text.json'),
                '3D_movements_text_vision': os.path.join(root_dir, '02_spatial', '3D_movements', 'urbansas_audio_bench_questions_text_vision.json'),

                'panaroma_audio_vision': os.path.join(root_dir, '02_spatial', 'panaroma', 'starss23_audio_bench_questions_audio_video.json'),
                'panaroma_vision_audio': os.path.join(root_dir, '02_spatial', 'panaroma', 'starss23_audio_bench_questions_video_audio.json'),
                'panaroma_audio_text': os.path.join(root_dir, '02_spatial', 'panaroma', 'starss23_audio_bench_questions_audio_text.json'),
                'panaroma_text_audio': os.path.join(root_dir, '02_spatial', 'panaroma', 'starss23_audio_bench_questions_text_audio.json'),
                'panaroma_vision_text': os.path.join(root_dir, '02_spatial', 'panaroma', 'starss23_audio_bench_questions_video_text.json'),
                'panaroma_text_vision': os.path.join(root_dir, '02_spatial', 'panaroma', 'starss23_audio_bench_questions_text_video.json'),
            },
            'speech': {
                'recognition_audio_vision': os.path.join(root_dir, '03_speech', 'recognition', 'acr_audio_bench_questions_audio_vision.json'),
                'recognition_vision_audio': os.path.join(root_dir, '03_speech', 'recognition', 'acr_audio_bench_questions_vision_audio.json'),
                'recognition_audio_text': os.path.join(root_dir, '03_speech', 'recognition', 'acr_audio_bench_questions_audio_text.json'),
                'recognition_text_audio': os.path.join(root_dir, '03_speech', 'recognition', 'acr_audio_bench_questions_text_audio.json'),
                'recognition_vision_text': os.path.join(root_dir, '03_speech', 'recognition', 'acr_audio_bench_questions_vision_text.json'),
                'recognition_text_vision': os.path.join(root_dir, '03_speech', 'recognition', 'acr_audio_bench_questions_text_vision.json'),

                'translation_audio_text': os.path.join(root_dir, '03_speech', 'translation', 'acr_audio_bench_questions_audio_text.json'),
                'translation_text_audio': os.path.join(root_dir, '03_speech', 'translation', 'acr_audio_bench_questions_text_audio.json'),
                'translation_vision_text': os.path.join(root_dir, '03_speech', 'translation', 'acr_audio_bench_questions_vision_text.json'),
                'translation_text_vision': os.path.join(root_dir, '03_speech', 'translation', 'acr_audio_bench_questions_text_vision.json'),
                'translation_vision_audio': os.path.join(root_dir, '03_speech', 'translation', 'acr_audio_bench_questions_vision_audio.json'),
                'translation_audio_vision': os.path.join(root_dir, '03_speech', 'translation', 'acr_audio_bench_questions_audio_vision.json'),
            
            },
            'temporal': {
                'count_audio_text': os.path.join(root_dir, '04_temporal', 'count', 'countixav_audio_bench_questions_audio_text.json'),
                'count_text_audio': os.path.join(root_dir, '04_temporal', 'count', 'countixav_audio_bench_questions_text_audio.json'),
                'count_vision_text': os.path.join(root_dir, '04_temporal', 'count', 'countixav_audio_bench_questions_vision_text.json'),
                'count_text_vision': os.path.join(root_dir, '04_temporal', 'count', 'countixav_audio_bench_questions_text_vision.json'),
                'count_audio_vision': os.path.join(root_dir, '04_temporal', 'count', 'countixav_audio_bench_questions_audio_vision.json'),
                'count_vision_audio': os.path.join(root_dir, '04_temporal', 'count', 'countixav_audio_bench_questions_vision_audio.json'),

                'calculation_audio_text': os.path.join(root_dir, '04_temporal', 'calculation', 'countixav_audio_bench_questions_audio_text.json'),
                'calculation_text_audio': os.path.join(root_dir, '04_temporal', 'calculation', 'countixav_audio_bench_questions_text_audio.json'),
                'calculation_vision_text': os.path.join(root_dir, '04_temporal', 'calculation', 'countixav_audio_bench_questions_vision_text.json'),
                'calculation_text_vision': os.path.join(root_dir, '04_temporal', 'calculation', 'countixav_audio_bench_questions_text_vision.json'),
                'calculation_audio_vision': os.path.join(root_dir, '04_temporal', 'calculation', 'countixav_audio_bench_questions_audio_vision.json'),
                'calculation_vision_audio': os.path.join(root_dir, '04_temporal', 'calculation', 'countixav_audio_bench_questions_vision_audio.json'),
            
                'order_audio_text': os.path.join(root_dir, '04_temporal', 'order', 'vggss_order_audio_bench_questions_audio_text.json'),
                'order_text_audio': os.path.join(root_dir, '04_temporal', 'order', 'vggss_order_audio_bench_questions_text_audio.json'),
                'order_vision_text': os.path.join(root_dir, '04_temporal', 'order', 'vggss_order_audio_bench_questions_vision_text.json'),
                'order_text_vision': os.path.join(root_dir, '04_temporal', 'order', 'vggss_order_audio_bench_questions_text_vision.json'),
                'order_audio_vision': os.path.join(root_dir, '04_temporal', 'order', 'vggss_order_audio_bench_questions_audio_vision.json'),
                'order_vision_audio': os.path.join(root_dir, '04_temporal', 'order', 'vggss_order_audio_bench_questions_vision_audio.json'),    
            },
            'external': {
                'music_genre_classification_audio_vision': os.path.join(root_dir, '05_Exteral', 'music_genre_classification', 'gtzan_music_questions_audio_vision.json'),
                'music_genre_classification_vision_audio': os.path.join(root_dir, '05_Exteral', 'music_genre_classification', 'gtzan_music_questions_vision_audio.json'),
                'music_genre_classification_audio_text': os.path.join(root_dir, '05_Exteral', 'music_genre_classification', 'gtzan_music_questions_audio_text.json'),
                'music_genre_classification_text_audio': os.path.join(root_dir, '05_Exteral', 'music_genre_classification', 'gtzan_music_questions_text_audio.json'),
                'music_genre_classification_vision_text': os.path.join(root_dir, '05_Exteral', 'music_genre_classification', 'gtzan_music_questions_vision_text.json'),
                'music_genre_classification_text_vision': os.path.join(root_dir, '05_Exteral', 'music_genre_classification', 'gtzan_music_questions_text_vision.json'),

                'emotion_classification_audio_vision': os.path.join(root_dir, '05_Exteral', 'emotion_classification', 'emotion_classification_questions_audio_vision.json'),
                'emotion_classification_vision_audio': os.path.join(root_dir, '05_Exteral', 'emotion_classification', 'emotion_classification_questions_vision_audio.json'),
                'emotion_classification_audio_text': os.path.join(root_dir, '05_Exteral', 'emotion_classification', 'emotion_classification_questions_audio_text.json'),
                'emotion_classification_text_audio': os.path.join(root_dir, '05_Exteral', 'emotion_classification', 'emotion_classification_questions_text_audio.json'),
                'emotion_classification_vision_text': os.path.join(root_dir, '05_Exteral', 'emotion_classification', 'emotion_classification_questions_vision_text.json'),
                'emotion_classification_text_vision': os.path.join(root_dir, '05_Exteral', 'emotion_classification', 'emotion_classification_questions_text_vision.json'),

                'movie_matching_audio_vision': os.path.join(root_dir, '05_Exteral', 'movie_matching', 'movie_matching_questions_audio_vision.json'),
                'movie_matching_vision_audio': os.path.join(root_dir, '05_Exteral', 'movie_matching', 'movie_matching_questions_vision_audio.json'),
                'movie_matching_audio_text': os.path.join(root_dir, '05_Exteral', 'movie_matching', 'movie_matching_questions_audio_text.json'),
                'movie_matching_text_audio': os.path.join(root_dir, '05_Exteral', 'movie_matching', 'movie_matching_questions_text_audio.json'),
                'movie_matching_vision_text': os.path.join(root_dir, '05_Exteral', 'movie_matching', 'movie_matching_questions_vision_text.json'),
                'movie_matching_text_vision': os.path.join(root_dir, '05_Exteral', 'movie_matching', 'movie_matching_questions_text_vision.json'),
                
                'singer_identification_audio_text': os.path.join(root_dir, '05_Exteral', 'singer_identification', 'singer_identification_questions_audio_text.json'),
                'singer_identification_text_audio': os.path.join(root_dir, '05_Exteral', 'singer_identification', 'singer_identification_questions_text_audio.json'),
                'singer_identification_vision_text': os.path.join(root_dir, '05_Exteral', 'singer_identification', 'singer_identification_questions_vision_text.json'),
                'singer_identification_vision_audio': os.path.join(root_dir, '05_Exteral', 'singer_identification', 'singer_identification_questions_vision_audio.json'),
                'singer_identification_audio_vision': os.path.join(root_dir, '05_Exteral', 'singer_identification', 'singer_identification_questions_audio_vision.json'),
                'singer_identification_text_vision': os.path.join(root_dir, '05_Exteral', 'singer_identification', 'singer_identification_questions_text_vision.json'),
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
    

