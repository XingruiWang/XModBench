import json
import statistics
import sys
import os

# Configuration
FILE_PATH = '/home/xwang378/scratch/2025/AudioBench/lite.json'

TASK_FAMILIES = {
    'perception': 'Perc.',
    'spatial': 'Spat.',
    'temporal': 'Temp.',
    'speech': 'Ling.',
    'external': 'Knwl.'
}

MODALITIES = {
    'audio_text': 'A->T',
    'audio_vision': 'A->V',
    'text_audio': 'T->A',
    'text_vision': 'T->V',
    'vision_audio': 'V->A',
    'vision_text': 'V->T'
}

# Enforce order for columns
TASK_ORDER = ['perception', 'spatial', 'temporal', 'speech', 'external']
MODALITY_ORDER = ['audio_text', 'audio_vision', 'text_audio', 'text_vision', 'vision_audio', 'vision_text']

# Load data
def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
        
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # Join lines and find the start of the JSON object (first '{')
    content = ''.join(lines)
    start_idx = content.find('{')
    if start_idx == -1:
        print("Error: No JSON object found in file")
        sys.exit(1)
        
    json_content = content[start_idx:]
    try:
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)

def main():
    data = load_data(FILE_PATH)
    
    # Define column headers
    task_headers = [TASK_FAMILIES[k] for k in TASK_ORDER]
    modality_headers = [MODALITIES[k] for k in MODALITY_ORDER]
    
    # Print Table Header
    # Adjust widths: Model (25), Tasks (7 each), Modalities (7 each), Std (8), Avg (8)
    header = f"{'Model':<25} |"
    for h in task_headers:
        header += f" {h:>6}"
    header += " |"
    for h in modality_headers:
        header += f" {h:>6}"
    header += " |   Std. |   Avg."
    
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    # Calculate metrics for each model
    model_metrics = []
    
    for model_name, model_data in data.items():
        # Aggregators
        task_family_scores = {k: [] for k in TASK_ORDER}
        modality_scores = {k: [] for k in MODALITY_ORDER}
        all_scores = []
        
        for task_family, tasks in model_data.items():
            if task_family not in TASK_ORDER:
                continue
                
            for task_name, modalities in tasks.items():
                if isinstance(modalities, dict):
                    for modality, score in modalities.items():
                        if modality in MODALITY_ORDER and isinstance(score, (int, float)):
                            task_family_scores[task_family].append(score)
                            modality_scores[modality].append(score)
                            all_scores.append(score)
        
        # Calculate Averages
        row_data = {'name': model_name}
        
        # Task Families
        for tf_key in TASK_ORDER:
            scores = task_family_scores[tf_key]
            row_data[TASK_FAMILIES[tf_key]] = statistics.mean(scores) if scores else 0.0
            
        # Modalities
        for mod_key in MODALITY_ORDER:
            scores = modality_scores[mod_key]
            row_data[MODALITIES[mod_key]] = statistics.mean(scores) if scores else 0.0
            
        # Overall
        if all_scores:
            row_data['Avg'] = statistics.mean(all_scores)
            row_data['Std'] = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
        else:
            row_data['Avg'] = 0.0
            row_data['Std'] = 0.0
            
        model_metrics.append(row_data)
        
    # Sort by Average score descending
    model_metrics.sort(key=lambda x: x['Avg'], reverse=True)
    
    # Print Rows
    for m in model_metrics:
        row_str = f"{m['name']:<25} |"
        
        for h in task_headers:
            row_str += f" {m[h]:6.1f}"
            
        row_str += " |"
        
        for h in modality_headers:
            row_str += f" {m[h]:6.1f}"
            
        row_str += f" | {m['Std']:6.4f} | {m['Avg']:6.1f}"
        
        print(row_str)

if __name__ == '__main__':
    main()
