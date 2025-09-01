import json
import os

class AudioBench():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        self.task_paths = {
            'perception': {
                'general_audio_vision': os.path.join(root_dir, '01_perception', 'general_activities', 'vggss_audio_bench_questions_audio_vision.json'),
                'general_vision_audio': os.path.join(root_dir, '01_perception', 'general_activities', 'vggss_audio_bench_questions_vision_audio.json'),
                'general_audio_text': os.path.join(root_dir, '01_perception', 'general_activities', 'vggss_audio_bench_questions_audio_text.json'),
                'general_vision_text': os.path.join(root_dir, '01_perception', 'general_activities', 'vggss_audio_bench_questions_vision_text.json'),
                'general_text_audio': os.path.join(root_dir, '01_perception', 'general_activities', 'vggss_audio_bench_questions_text_audio.json'),
                'general_text_vision': os.path.join(root_dir, '01_perception', 'general_activities', 'vggss_audio_bench_questions_text_vision.json'),
                
                'finegrained_audio_vision': os.path.join(root_dir, '01_perception', 'Finegrained', 'vggss_audio_bench_questions_audio_vision.json'),
                'finegrained_audio_text': os.path.join(root_dir, '01_perception', 'Finegrained', 'vggss_audio_bench_questions_audio_text.json'),
                'finegrained_text_audio': os.path.join(root_dir, '01_perception', 'Finegrained', 'vggss_audio_bench_questions_text_audio.json'),
                'finegrained_text_vision': os.path.join(root_dir, '01_perception', 'Finegrained', 'vggss_audio_bench_questions_text_vision.json'),
                'finegrained_vision_audio': os.path.join(root_dir, '01_perception', 'Finegrained', 'vggss_audio_bench_questions_vision_audio.json'),
                'finegrained_vision_text': os.path.join(root_dir, '01_perception', 'Finegrained', 'vggss_audio_bench_questions_vision_text.json'),
                
                'solos_audio_vision': os.path.join(root_dir, '01_perception', 'instruments', 'solos_audio_bench_questions_audio_vision.json'),
                'solos_vision_audio': os.path.join(root_dir, '01_perception', 'instruments', 'solos_audio_bench_questions_vision_audio.json'),
                'solos_vision_text': os.path.join(root_dir, '01_perception', 'instruments', 'solos_audio_bench_questions_vision_text.json'),
                'solos_audio_text': os.path.join(root_dir, '01_perception', 'instruments', 'solos_audio_bench_questions_audio_text.json'),
                'solos_text_audio': os.path.join(root_dir, '01_perception', 'instruments', 'solos_audio_bench_questions_text_audio.json'),
                'solos_text_vision': os.path.join(root_dir, '01_perception', 'instruments', 'solos_audio_bench_questions_text_vision.json'),
                
                'urmp_audio_vision': os.path.join(root_dir, '01_perception', 'instruments_comp', 'URMP_audio_bench_questions_audio_vision.json'),
                'urmp_vision_audio': os.path.join(root_dir, '01_perception', 'instruments_comp', 'URMP_audio_bench_questions_vision_audio.json'),
                'urmp_audio_text': os.path.join(root_dir, '01_perception', 'instruments_comp', 'URMP_audio_bench_questions_audio_text.json'),
                'urmp_text_audio': os.path.join(root_dir, '01_perception', 'instruments_comp', 'URMP_audio_bench_questions_text_audio.json'),
                'urmp_vision_text': os.path.join(root_dir, '01_perception', 'instruments_comp', 'URMP_audio_bench_questions_vision_text.json'),
                'urmp_text_vision': os.path.join(root_dir, '01_perception', 'instruments_comp', 'URMP_audio_bench_questions_text_vision.json'),
                
                'landscapes_audio_vision': os.path.join(root_dir, '01_perception', 'natures', 'landscapes_audio_bench_questions_audio_vision.json'),
                'landscapes_vision_audio': os.path.join(root_dir, '01_perception', 'natures', 'landscapes_audio_bench_questions_vision_audio.json'),
                'landscapes_audio_text': os.path.join(root_dir, '01_perception', 'natures', 'landscapes_audio_bench_questions_audio_text.json'),
                'landscapes_text_audio': os.path.join(root_dir, '01_perception', 'natures', 'landscapes_audio_bench_questions_text_audio.json'),
                'landscapes_vision_text': os.path.join(root_dir, '01_perception', 'natures', 'landscapes_audio_bench_questions_vision_text.json'),
                'landscapes_text_vision': os.path.join(root_dir, '01_perception', 'natures', 'landscapes_audio_bench_questions_text_vision.json'),
            },
            
            'spatial': {
                'urmp_audio_vision': os.path.join(root_dir, '02_spatial', 'urmp', 'urmp_audio_bench_questions_audio_vision.json'),
                'urmp_vision_audio': os.path.join(root_dir, '02_spatial', 'urmp', 'urmp_audio_bench_questions_vision_audio.json'),
                'urmp_audio_text': os.path.join(root_dir, '02_spatial', 'urmp', 'urmp_audio_bench_questions_audio_text.json'),
                'urmp_text_audio': os.path.join(root_dir, '02_spatial', 'urmp', 'urmp_audio_bench_questions_text_audio.json'),
                'urmp_vision_text': os.path.join(root_dir, '02_spatial', 'urmp', 'urmp_audio_bench_questions_vision_text.json'),
                'urmp_text_vision': os.path.join(root_dir, '02_spatial', 'urmp', 'urmp_audio_bench_questions_text_vision.json'),
            },
            'spatial_easy': {
                'urmp_audio_vision': os.path.join(root_dir, '02_spatial', 'urmp_easy', 'urmp_audio_bench_questions_audio_vision.json'),
                'urmp_vision_audio': os.path.join(root_dir, '02_spatial', 'urmp_easy', 'urmp_audio_bench_questions_vision_audio.json'),
                'urmp_audio_text': os.path.join(root_dir, '02_spatial', 'urmp_easy', 'urmp_audio_bench_questions_audio_text.json'),
                'urmp_text_audio': os.path.join(root_dir, '02_spatial', 'urmp_easy', 'urmp_audio_bench_questions_text_audio.json'),
                'urmp_vision_text': os.path.join(root_dir, '02_spatial', 'urmp_easy', 'urmp_audio_bench_questions_vision_text.json'),
                'urmp_text_vision': os.path.join(root_dir, '02_spatial', 'urmp_easy', 'urmp_audio_bench_questions_text_vision.json'),
            },
            'spatial_outdoor': {
                'urbansas_audio_vision': os.path.join(root_dir, '02_spatial', '3D_movements', 'urbansas_audio_bench_questions_audio_vision.json'),
                'urbansas_vision_audio': os.path.join(root_dir, '02_spatial', '3D_movements', 'urbansas_audio_bench_questions_vision_audio.json'),
                'urbansas_audio_text': os.path.join(root_dir, '02_spatial', '3D_movements', 'urbansas_audio_bench_questions_audio_text.json'),
                'urbansas_text_audio': os.path.join(root_dir, '02_spatial', '3D_movements', 'urbansas_audio_bench_questions_text_audio.json'),
                'urbansas_vision_text': os.path.join(root_dir, '02_spatial', '3D_movements', 'urbansas_audio_bench_questions_vision_text.json'),
                'urbansas_text_vision': os.path.join(root_dir, '02_spatial', '3D_movements', 'urbansas_audio_bench_questions_text_vision.json'),
            },
            'spatial_indoor': {
                'starss23_audio_video': os.path.join(root_dir, '02_spatial', 'starss23', 'starss23_audio_bench_questions_audio_video.json'),
                'starss23_video_audio': os.path.join(root_dir, '02_spatial', 'starss23', 'starss23_audio_bench_questions_video_audio.json'),
                'starss23_audio_text': os.path.join(root_dir, '02_spatial', 'starss23', 'starss23_audio_bench_questions_audio_text.json'),
                'starss23_text_audio': os.path.join(root_dir, '02_spatial', 'starss23', 'starss23_audio_bench_questions_text_audio.json'),
                'starss23_video_text': os.path.join(root_dir, '02_spatial', 'starss23', 'starss23_audio_bench_questions_video_text.json'),
                'starss23_text_video': os.path.join(root_dir, '02_spatial', 'starss23', 'starss23_audio_bench_questions_text_video.json'),
            },
            'acr': {
                # 'acr_audio_vision': os.path.join(root_dir, '03_speech', 'acr', 'acr_audio_bench_questions_audio_vision.json'),
                # 'acr_vision_audio': os.path.join(root_dir, '03_speech', 'acr', 'acr_audio_bench_que tions_audio_text.json'),
                # 'acr_text_audio': os.path.join(root_dir, '03_speech', 'acr', 'acr_audio_bench_questions_text_audio.json'),
                # 'acr_vision_text': os.path.join(root_dir, '03_speech', 'acr', 'acr_audio_bench_questions_vision_text.json'),
                # 'acr_text_vision': os.path.join(root_dir, '03_speech', 'acr', 'acr_audio_bench_questions_text_vision.json'),
                
                'acr_hard_audio_vision': os.path.join(root_dir, '03_speech', 'recognition', 'acr_audio_bench_questions_audio_vision.json'),
                'acr_hard_vision_audio': os.path.join(root_dir, '03_speech', 'recognition', 'acr_audio_bench_questions_vision_audio.json'),
                'acr_hard_audio_text': os.path.join(root_dir, '03_speech', 'recognition', 'acr_audio_bench_questions_audio_text.json'),
                'acr_hard_text_audio': os.path.join(root_dir, '03_speech', 'recognition', 'acr_audio_bench_questions_text_audio.json'),
                'acr_hard_vision_text': os.path.join(root_dir, '03_speech', 'recognition', 'acr_audio_bench_questions_vision_text.json'),
                'acr_hard_text_vision': os.path.join(root_dir, '03_speech', 'recognition', 'acr_audio_bench_questions_text_vision.json'),
            },
            # 'ocr_translation_Chinese': {
            #     'ocr_translation_audio_text': os.path.join(root_dir, '03_ocr_translation', 'translation', 'acr_audio_bench_questions_audio_text.json'),
            #     'ocr_translation_text_audio': os.path.join(root_dir, '03_ocr_translation', 'translation', 'acr_audio_bench_questions_text_audio.json'),
            #     'ocr_translation_vision_text': os.path.join(root_dir, '03_ocr_translation', 'acr_Chinese', 'acr_audio_bench_questions_vision_text.json'),
            #     'ocr_translation_text_vision': os.path.join(root_dir, '03_ocr_translation', 'acr_Chinese', 'acr_audio_bench_questions_text_vision.json'),
            # },
            'acr_translation_Chinese_hard': {
                'ocr_translation_audio_text': os.path.join(root_dir, '03_speech', 'translation', 'acr_audio_bench_questions_audio_text.json'),
                'ocr_translation_text_audio': os.path.join(root_dir, '03_speech', 'translation', 'acr_audio_bench_questions_text_audio.json'),
                'ocr_translation_vision_text': os.path.join(root_dir, '03_speech', 'translation', 'acr_audio_bench_questions_vision_text.json'),
                'ocr_translation_vision_audio': os.path.join(root_dir, '03_speech', 'translation', 'acr_audio_bench_questions_vision_audio.json'),
            },
            'temporal_count': {
                'countixav_audio_text': os.path.join(root_dir, '04_temporal', 'countixav_Image', 'countixav_audio_bench_questions_audio_text.json'),
                'countixav_text_audio': os.path.join(root_dir, '04_temporal', 'countixav_Image', 'countixav_audio_bench_questions_text_audio.json'),
                'countixav_vision_text': os.path.join(root_dir, '04_temporal', 'countixav_Image', 'countixav_audio_bench_questions_vision_text.json'),
                'countixav_text_vision': os.path.join(root_dir, '04_temporal', 'countixav_Image', 'countixav_audio_bench_questions_text_vision.json'),
                'countixav_audio_vision': os.path.join(root_dir, '04_temporal', 'countixav_Image', 'countixav_audio_bench_questions_audio_vision.json'),
                'countixav_vision_audio': os.path.join(root_dir, '04_temporal', 'countixav_Image', 'countixav_audio_bench_questions_vision_audio.json'),
                
                'countixav_video_text': os.path.join(root_dir, '04_temporal', 'countixav_Video', 'countixav_audio_bench_questions_vision_text.json'),
                'countixav_text_video': os.path.join(root_dir, '04_temporal', 'countixav_Video', 'countixav_audio_bench_questions_text_vision.json'),
                'countixav_audio_video': os.path.join(root_dir, '04_temporal', 'countixav_Video', 'countixav_audio_bench_questions_audio_vision.json'),
                'countixav_video_audio': os.path.join(root_dir, '04_temporal', 'countixav_Video', 'countixav_audio_bench_questions_vision_audio.json'),
            },
            'temporal_count_reasoning': {
                'countixav_audio_text': os.path.join(root_dir, '04_temporal', 'countixav_Image_reasoning', 'countixav_audio_bench_questions_audio_text.json'),
                'countixav_text_audio': os.path.join(root_dir, '04_temporal', 'countixav_Image_reasoning', 'countixav_audio_bench_questions_text_audio.json'),
                'countixav_vision_text': os.path.join(root_dir, '04_temporal', 'countixav_Image_reasoning', 'countixav_audio_bench_questions_vision_text.json'),
                'countixav_text_vision': os.path.join(root_dir, '04_temporal', 'countixav_Image_reasoning', 'countixav_audio_bench_questions_text_vision.json'),
                'countixav_audio_vision': os.path.join(root_dir, '04_temporal', 'countixav_Image_reasoning', 'countixav_audio_bench_questions_audio_vision.json'),
                'countixav_vision_audio': os.path.join(root_dir, '04_temporal', 'countixav_Image_reasoning', 'countixav_audio_bench_questions_vision_audio.json'),
                
                'countixav_video_text': os.path.join(root_dir, '04_temporal', 'countixav_Video_reasoning', 'countixav_audio_bench_questions_vision_text.json'),
                'countixav_text_video': os.path.join(root_dir, '04_temporal', 'countixav_Video_reasoning', 'countixav_audio_bench_questions_text_vision.json'),
                'countixav_audio_video': os.path.join(root_dir, '04_temporal', 'countixav_Video_reasoning', 'countixav_audio_bench_questions_audio_vision.json'),
                'countixav_video_audio': os.path.join(root_dir, '04_temporal', 'countixav_Video_reasoning', 'countixav_audio_bench_questions_vision_audio.json'),
            },
            'temporal_order':{
                'vggss_order_audio_text': os.path.join(root_dir, '04_temporal', 'vggss_order_order', 'vggss_order_audio_bench_questions_audio_text.json'),
                'vggss_order_text_audio': os.path.join(root_dir, '04_temporal', 'vggss_order_order', 'vggss_order_audio_bench_questions_text_audio.json'),
                'vggss_order_vision_text': os.path.join(root_dir, '04_temporal', 'vggss_order_order', 'vggss_order_audio_bench_questions_vision_text.json'),
                'vggss_order_text_vision': os.path.join(root_dir, '04_temporal', 'vggss_order_order', 'vggss_order_audio_bench_questions_text_vision.json'),
                'vggss_order_audio_vision': os.path.join(root_dir, '04_temporal', 'vggss_order_order', 'vggss_order_audio_bench_questions_audio_vision.json'),
                'vggss_order_vision_audio': os.path.join(root_dir, '04_temporal', 'vggss_order_order', 'vggss_order_audio_bench_questions_vision_audio.json'),    
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
    

