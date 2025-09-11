import os
import json
import csv

TASKS_TO_RUN=[
    # Perception tasks
    "perception/general",
    "perception/finegrained",
    "perception/instruments",
    "perception/instruments_comp",
    "perception/natures",
    
    # Spatial tasks
    "spatial/arrangements",
    "spatial/3D_movements",
    "spatial/panaroma",
    
    # Speech tasks
    "speech/recognition",
    "speech/translation",
    
    # Temporal tasks
    "temporal/count",
    "temporal/calculation",
    "temporal/order",
    
    # External tasks
    "external/music_genre_classification",
    "external/emotion_classification",
    "external/movie_matching",
    "external/singer_identification"
]

modalities = ["audio_vision", "vision_audio", "vision_text", "text_vision", "text_audio", "audio_text"]


class BenchmarkMetric:
    def __init__(self, result_root, save_result_json):
        self.result_root = result_root
        self.save_result_json = save_result_json

    def analysis_all_model(self):
        result_dict = {}
        models = ['gemini-2.5-flash', 'gemini-2.0-pro', 'gemini-2.5-pro', 'gemini-2.0-flash', 'qwen2.5_omni', 'vita', 'echoink', 'anygpt']

        for model in models:
            if model.startswith(".") or model.startswith("_"):
                continue
            if model not in result_dict:
                result_dict[model] = {}
            for task in TASKS_TO_RUN:
                task_name, subtask = task.split("/")
                if task_name not in result_dict[model]:
                    result_dict[model][task_name] = {}
                if subtask not in result_dict[model][task_name]:
                    result_dict[model][task_name][subtask] = {}

                for modality in modalities:
                    result_file = os.path.join(self.result_root, model, f"{task_name}_{subtask}_{modality}.json")
                    if not os.path.exists(result_file):
                        mini_result_file = os.path.join(self.result_root.replace("results", "results_mini_benchmark"), model, f"{task_name}_{subtask}_{modality}.json")
                        if not os.path.exists(mini_result_file):
                            print(f"No result file for {task_name} {subtask} {modality} for model {model}")
                            continue
                        result_file = mini_result_file
                    with open(result_file, "r") as f:
                        result = json.load(f)
                    score = result["score"]
                    result_dict[model][task_name][subtask][modality] = score
        with open(self.save_result_json, "w") as f:
            json.dump(result_dict, f)

    def get_task_statistics(self):
        """
        Extract all_count statistics from result files for each task/modality combination
        """
        task_stats = {}
        
        # Get any model to extract statistics (assuming all models have same task structure)
        models = [m for m in os.listdir(self.result_root) if not m.startswith(".") and not m.startswith("_")]
        if not models:
            return task_stats
            
        # Use the first available model to get statistics
        sample_model = models[0]
        
        for task in TASKS_TO_RUN:
            task_name, subtask = task.split("/")
            if task_name not in task_stats:
                task_stats[task_name] = {}
            if subtask not in task_stats[task_name]:
                task_stats[task_name][subtask] = {}

            for modality in modalities:
                result_file = os.path.join(self.result_root, sample_model, f"{task_name}_{subtask}_{modality}.json")
                if os.path.exists(result_file):
                    try:
                        with open(result_file, "r") as f:
                            result = json.load(f)
                        # Extract all_count if it exists
                        if "all_count" in result:
                            task_stats[task_name][subtask][modality] = result["all_count"]
                        else:
                            task_stats[task_name][subtask][modality] = "N/A"
                    except (json.JSONDecodeError, KeyError):
                        task_stats[task_name][subtask][modality] = "N/A"
                else:
                    task_stats[task_name][subtask][modality] = "N/A"
        
        return task_stats

    def to_csv(self, csv_file='benchmark_results.csv'):
        """
        Export benchmark results to CSV format matching the spreadsheet layout
        
        Args:
            csv_file: Output CSV filename
        """
        
        # Define the header rows based on the image structure
        headers = [
            'Model',
            'Interleaved Modality',
            'T1 (Perception)',
            '',
            '',
            '',
            '',
            'T2 Spatial',
            '',
            '',
            'T3 Temporal',
            '',
            '',
            'T4 Speech',
            '',
            'T5 External',
            '',
            '',
            ''
        ]
        
        sub_headers = [
            '',
            '',
            'General',
            'General - Hard',  # finegrained
            'Scene',           # natures
            'Instruments',
            'Instruments-multi',  # instruments_comp
            'Arrangement',     # arrangements
            'Moving Direction', # 3D_movements
            'Indoor',          # panaroma
            'Order',
            'Counting',        # count
            'Calculation',
            'Recognition',     # speech/recognition
            'Translation',     # speech/translation
            'Genre',           # music_genre_classification
            'Emotion',         # emotion_classification
            'Movie',           # movie_matching
            'Singer'           # singer_identification
        ]

        # Map the modality names to match the spreadsheet format
        modality_map = {
            'audio_vision': 'Audio -> Vision',
            'vision_audio': 'Vision -> Audio', 
            'vision_text': 'Vision -> Text',
            'text_vision': 'Text - Vision',
            'text_audio': 'Text -> Audio',
            'audio_text': 'Audio -> Text'
        }

        # Map task/subtask combinations to column indices (after Model and Modality columns)
        task_column_map = {
            ('perception', 'general'): 2,
            ('perception', 'finegrained'): 3,
            ('perception', 'natures'): 4,
            ('perception', 'instruments'): 5,
            ('perception', 'instruments_comp'): 6,
            ('spatial', 'arrangements'): 7,
            ('spatial', '3D_movements'): 8,
            ('spatial', 'panaroma'): 9,
            ('temporal', 'order'): 10,
            ('temporal', 'count'): 11,
            ('temporal', 'calculation'): 12,
            ('speech', 'recognition'): 13,
            ('speech', 'translation'): 14,
            ('external', 'music_genre_classification'): 15,
            ('external', 'emotion_classification'): 16,
            ('external', 'movie_matching'): 17,
            ('external', 'singer_identification'): 18
        }

        # Load the result data
        with open(self.save_result_json, 'r') as f:
            result_dict = json.load(f)

        # Get task statistics
        task_stats = self.get_task_statistics()

        # Prepare CSV data
        csv_data = []
        
        # Add statistics row at the top
        stats_row = ['Statistics (all_count)', ''] + [''] * (len(headers) - 2)
        for task_name, task_data in task_stats.items():
            for subtask_name, subtask_data in task_data.items():
                column_idx = task_column_map.get((task_name, subtask_name))
                if column_idx is not None:
                    # Get the first available modality's count (assuming counts are same across modalities)
                    for modality, count in subtask_data.items():
                        if count != "N/A":
                            stats_row[column_idx] = str(count)
                            break
                    else:
                        stats_row[column_idx] = "N/A"
        
        csv_data.append(stats_row)
        csv_data.append(headers)
        csv_data.append(sub_headers)

        # Process each model
        for model_name, model_data in result_dict.items():
            # Format model name (replace hyphens and capitalize)
            model_display_name = model_name.replace('-', ' ').replace('_', ' ').title()
            
            # Get all unique modalities across all tasks for this model
            all_modalities = set()
            for task_name, task_data in model_data.items():
                for subtask_name, subtask_data in task_data.items():
                    if isinstance(subtask_data, dict):
                        all_modalities.update(subtask_data.keys())

            # Process each modality
            for i, modality in enumerate(sorted(all_modalities)):
                # Initialize row with empty values
                row = [''] * len(headers)
                
                # Model name (only on first row for this model)
                if i == 0:
                    row[0] = model_display_name
                
                # Interleaved Modality
                row[1] = modality_map.get(modality, modality)
                
                # Fill in the scores for each task/subtask combination
                for task_name, task_data in model_data.items():
                    for subtask_name, subtask_data in task_data.items():
                        if isinstance(subtask_data, dict) and modality in subtask_data:
                            # Get the column index for this task/subtask
                            column_idx = task_column_map.get((task_name, subtask_name))
                            if column_idx is not None:
                                score = subtask_data[modality]
                                # Format score (round to 2 decimal places if it's a number)
                                if isinstance(score, (int, float)):
                                    row[column_idx] = f"{score:.2f}" if score != int(score) else str(int(score))
                                else:
                                    row[column_idx] = str(score)

                
                csv_data.append(row)

        # Write to CSV file
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)
        
        print(f"CSV file '{csv_file}' has been created successfully!")
        print(f"Exported results for {len(result_dict)} models")
        print(f"Added statistics row showing all_count for each task")
        
        # Print preview
        print("\nCSV Preview:")
        print("-" * 120)
        for i, row in enumerate(csv_data[:10]):  # Show first few rows including stats
            print(','.join(str(cell)[:10] for cell in row))  # Truncate long values for display
            if i == 0 or i == 2:  # Add separator after stats row and sub-headers
                print("-" * 120)
        
        return csv_data



if __name__ == "__main__":
    benchmark_metric = BenchmarkMetric(
        result_root="/home/xwang378/scratch/2025/AudioBench/benchmark/results",
        save_result_json="/home/xwang378/scratch/2025/AudioBench/benchmark/all_model_result.json"
    )
    benchmark_metric.analysis_all_model()
    csv_data = benchmark_metric.to_csv()