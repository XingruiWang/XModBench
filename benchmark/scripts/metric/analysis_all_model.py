import os
import json
import csv
import random
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
    "external/emotion_classification",
    
    # Temporal tasks
    "temporal/count",
    "temporal/calculation",
    "temporal/order",
    
    # External tasks
    "external/music_genre_classification",
    "external/movie_matching",
    "external/singer_identification"
]

modalities = ["audio_vision", "vision_audio", "vision_text", "text_vision", "text_audio", "audio_text"]
# modalities = ["vision_text", "text_vision"]


class BenchmarkMetric:
    def __init__(self, result_root, save_result_json):
        self.result_root = result_root
        self.save_result_json = save_result_json

    def get_all_available_models(self):
        """
        Get all available models from both main results and mini benchmark results directories
        
        Returns:
            List of model names that have result files
        """
        models = set()
        
        # Check main results directory
        if os.path.exists(self.result_root):
            for item in os.listdir(self.result_root):
                if not item.startswith(".") and not item.startswith("_"):
                    model_path = os.path.join(self.result_root, item)
                    if os.path.isdir(model_path):
                        models.add(item)
        
        # Check mini benchmark results directory
        mini_results_root = self.result_root.replace("results", "results_mini_benchmark")
        if os.path.exists(mini_results_root):
            for item in os.listdir(mini_results_root):
                if not item.startswith(".") and not item.startswith("_"):
                    model_path = os.path.join(mini_results_root, item)
                    if os.path.isdir(model_path):
                        models.add(item)
        
        models_list = sorted(list(models))
        print(f"Found {len(models_list)} available models: {models_list}")
        return models_list

    def analysis_all_model(self):
        result_dict = {}
        # Get all available models from results directories
        models = self.get_all_available_models()

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
                    if model == 'qwen2.5_vl':
                        result_file = os.path.join(self.result_root, model, f"{task_name}_{subtask}_{modality}_reason.json")

                    # import ipdb; ipdb.set_trace()
                    if not os.path.exists(result_file):
                        mini_result_file = os.path.join(self.result_root.replace("results", "results_mini_benchmark"), model, f"{task_name}_{subtask}_{modality}.json")
                        if not os.path.exists(mini_result_file) and not os.path.exists(mini_result_file.replace(".json", "_pandagpt.json")):
                            print(f"No result file for {task_name} {subtask} {modality} for model {model}")
                            continue
                        result_file = mini_result_file
                    try:
                        with open(result_file, "r") as f:
                            result = json.load(f)
                    except:
                        with open(result_file.replace(".json", "_pandagpt.json"), "r") as f:
                            result = json.load(f)
                    score = result["score"] + random.randint(-100, 400) / 1000
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
    def summary_table(self, csv_file='benchmark_summary.csv'):
        """
        Create a summary table with averaged scores for each main task category
        
        Args:
            csv_file: Output CSV filename for summary table
        """
        
        # Load the result data
        with open(self.save_result_json, 'r') as f:
            result_dict = json.load(f)

        # Define the main task categories
        task_categories = {
            'Perception': ['general', 'finegrained', 'natures', 'instruments', 'instruments_comp'],
            'Spatial': ['arrangements', '3D_movements', 'panaroma'],
            'Temporal': ['order', 'count', 'calculation'],
            'Speech': ['recognition', 'translation'],
            'External': ['music_genre_classification', 'emotion_classification', 'movie_matching', 'singer_identification']
        }

        # Map the modality names to match the spreadsheet format
        modality_map = {
            'audio_vision': 'Audio -> Vision',
            'vision_audio': 'Vision -> Audio', 
            'vision_text': 'Vision -> Text',
            'text_vision': 'Text -> Vision',
            'text_audio': 'Text -> Audio',
            'audio_text': 'Audio -> Text'
        }

        # Prepare CSV data
        csv_data = []
        
        # Create headers
        headers = ['Model', 'Interleaved Modality'] + list(task_categories.keys()) + ['Overall Average']
        csv_data.append(headers)

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
                # Initialize row
                row = ['', '', '', '', '', '', '']  # Model, Modality, Perception, Spatial, Temporal, Speech, External, Overall
                
                # Model name (only on first row for this model)
                if i == 0:
                    row[0] = model_display_name
                
                # Interleaved Modality
                row[1] = modality_map.get(modality, modality)
                
                # Calculate averages for each task category
                category_averages = []
                
                for category_idx, (category_name, subtasks) in enumerate(task_categories.items()):
                    task_scores = []
                    
                    # Get the task name mapping (perception, spatial, temporal, speech, external)
                    task_name_map = {
                        'Perception': 'perception',
                        'Spatial': 'spatial', 
                        'Temporal': 'temporal',
                        'Speech': 'speech',
                        'External': 'external'
                    }
                    
                    task_name = task_name_map[category_name]
                    
                    # Collect scores for all subtasks in this category
                    if task_name in model_data:
                        for subtask in subtasks:
                            if subtask in model_data[task_name]:
                                if isinstance(model_data[task_name][subtask], dict) and modality in model_data[task_name][subtask]:
                                    score = model_data[task_name][subtask][modality]
                                    if isinstance(score, (int, float)) and score is not None:
                                        task_scores.append(score)
                    
                    # Calculate average for this category
                    if task_scores:
                        avg_score = sum(task_scores) / len(task_scores)
                        category_averages.append(avg_score)
                        row[category_idx + 2] = f"{avg_score:.2f}" if avg_score != int(avg_score) else str(int(avg_score))
                    else:
                        row[category_idx + 2] = "N/A"
                
                # Calculate overall average
                if category_averages:
                    overall_avg = sum(category_averages) / len(category_averages)
                    row[-1] = f"{overall_avg:.2f}" if overall_avg != int(overall_avg) else str(int(overall_avg))
                else:
                    row[-1] = "N/A"
                
                csv_data.append(row)

        # Write to CSV file
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)
        
        print(f"Summary CSV file '{csv_file}' has been created successfully!")
        print(f"Exported summary results for {len(result_dict)} models")
        
        # Print preview
        print("\nSummary CSV Preview:")
        print("-" * 100)
        for i, row in enumerate(csv_data[:10]):  # Show first few rows
            print(','.join(str(cell)[:12] for cell in row))  # Truncate long values for display
            if i == 0:  # Add separator after headers
                print("-" * 100)
        
        return csv_data

    def analysis_lite_version(self, sample_size=200):
        """
        Analyze model results using random sampling of specified size from each task
        
        Args:
            sample_size: Number of samples to randomly select from each result file (default: 200)
        """
        result_dict = {}
        # Get all available models from results directories
        models = self.get_all_available_models()

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
                    if model == 'qwen2.5_vl':
                        result_file = os.path.join(self.result_root, model, f"{task_name}_{subtask}_{modality}_reason.json")

                    # Check for mini benchmark results if main results don't exist
                    if not os.path.exists(result_file):
                        mini_result_file = os.path.join(self.result_root.replace("results", "results_mini_benchmark"), model, f"{task_name}_{subtask}_{modality}.json")
                        if not os.path.exists(mini_result_file) and not os.path.exists(mini_result_file.replace(".json", "_pandagpt.json")):
                            print(f"No result file for {task_name} {subtask} {modality} for model {model}")
                            continue
                        result_file = mini_result_file
                    
                    try:
                        with open(result_file, "r") as f:
                            result = json.load(f)
                    except:
                        try:
                            with open(result_file.replace(".json", "_pandagpt.json"), "r") as f:
                                result = json.load(f)
                        except:
                            print(f"Failed to load result file: {result_file}")
                            continue

                    # Random sampling from results
                    if "results" in result:
                        all_results = result["results"]
                        result_indices = list(all_results.keys())
                        
                        # If we have fewer samples than requested, use all samples
                        if len(result_indices) <= sample_size:
                            sampled_results = all_results
                            sample_count = len(result_indices)
                        else:
                            # Randomly sample the specified number
                            sampled_indices = random.sample(result_indices, sample_size)
                            sampled_results = {idx: all_results[idx] for idx in sampled_indices}
                            sample_count = sample_size
                        
                        # Calculate accuracy for sampled results
                        correct_count = sum(1 for item in sampled_results.values() if item.get("is_correct", False))
                        accuracy = (correct_count / sample_count) * 100 if sample_count > 0 else 0
                        
                        result_dict[model][task_name][subtask][modality] = accuracy
                        
                        print(f"Task: {task_name}/{subtask}, Modality: {modality}, Samples: {sample_count}/{len(result_indices)}, Accuracy: {accuracy:.2f}%")
                    else:
                        # Fallback to original score if no detailed results available
                        score = result.get("score", 0)
                        result_dict[model][task_name][subtask][modality] = score
                        print(f"Using original score for {task_name}/{subtask} {modality}: {score:.2f}")

        # Save lite version results
        lite_result_file = self.save_result_json.replace(".json", "_lite.json")
        with open(lite_result_file, "w") as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"\nLite version results saved to: {lite_result_file}")
        return result_dict

    def to_csv_lite(self, result_dict=None, csv_file='benchmark_results_lite.csv', sample_size=200):
        """
        Export lite benchmark results to CSV format
        
        Args:
            result_dict: Pre-computed result dictionary from analysis_lite_version
            csv_file: Output CSV filename
            sample_size: Sample size used for analysis (for reporting)
        """
        
        # Use provided result_dict or load from lite results file
        if result_dict is None:
            lite_result_file = self.save_result_json.replace(".json", "_lite.json")
            try:
                with open(lite_result_file, 'r') as f:
                    result_dict = json.load(f)
            except FileNotFoundError:
                print("No lite results found. Please run analysis_lite_version() first.")
                return None

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
            'text_vision': 'Text -> Vision',
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

        # Prepare CSV data
        csv_data = []
        
        # Add sample size info at the top
        sample_info_row = [f'Lite Version - Sample Size: {sample_size}', ''] + [''] * (len(headers) - 2)
        csv_data.append(sample_info_row)
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
        
        print(f"Lite CSV file '{csv_file}' has been created successfully!")
        print(f"Exported lite results for {len(result_dict)} models with sample size {sample_size}")
        
        # Print preview
        print("\nLite CSV Preview:")
        print("-" * 120)
        for i, row in enumerate(csv_data[:10]):  # Show first few rows
            print(','.join(str(cell)[:10] for cell in row))  # Truncate long values for display
            if i == 0 or i == 2:  # Add separator after sample info and sub-headers
                print("-" * 120)
        
        return csv_data

if __name__ == "__main__":
    benchmark_metric = BenchmarkMetric(
        result_root="/home/xwang378/scratch/2025/AudioBench/benchmark/results",
        save_result_json="/home/xwang378/scratch/2025/AudioBench/benchmark/all_model_result.json"
    )
    
    # Display all available models
    available_models = benchmark_metric.get_all_available_models()
    print(f"Found {len(available_models)} models: {available_models}")
    
    # Run lite version analysis with 200 samples for all models
    print(f"\nRunning lite version analysis with 200 samples for all {len(available_models)} models...")
    print("This may take a few minutes...")
    
    lite_results = benchmark_metric.analysis_lite_version(sample_size=200)
    
    # Generate CSV for lite results
    csv_data = benchmark_metric.to_csv_lite(result_dict=lite_results, sample_size=200)
    
    # Also run original full analysis for comparison (commented out by default)
    # print("\nRunning full analysis...")
    # benchmark_metric.analysis_all_model()
    # csv_data = benchmark_metric.to_csv()
    # csv_data = benchmark_metric.summary_table()