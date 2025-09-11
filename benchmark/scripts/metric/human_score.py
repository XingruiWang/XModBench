import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

def extract_modality_from_question_type(question_type):
    """Extract modality from question type string and convert to arrow format"""
    modality_mapping = {
        'text_audio': 'Text -> Audio',
        'vision_audio': 'Vision -> Audio', 
        'text_vision': 'Text - Vision',
        'audio_text': 'Audio -> Text',
        'vision_text': 'Vision -> Text',
        'audio_vision': 'Audio -> Vision'
    }
    
    for key, value in modality_mapping.items():
        if key in question_type.lower():
            return value
    
    return "Unknown"

def extract_task_subtask_info(data, filename):
    """Extract task and subtask information"""
    
    # Try to get from JSON first
    task = data.get('task', '')
    subtask = data.get('subtask', '')
    
    # If not in JSON, parse from filename
    if not task or not subtask:
        parts = filename.replace('.json', '').split('_')
        
        # Look for task pattern (number_taskname)
        for i in range(len(parts)-1):
            if parts[i].isdigit() and len(parts[i]) == 2:
                task_num = parts[i]
                if i+1 < len(parts):
                    task_name = parts[i+1]
                    task = f"{task_num}_{task_name}"
                if i+2 < len(parts):
                    subtask = parts[i+2]
                break
    
    return task, subtask

def process_all_user_files_and_create_csv(input_directory, output_csv="human_results_table.csv"):
    """Process all JSON files and create CSV in the exact format shown"""
    
    all_data = []
    json_files = list(Path(input_directory).glob('*.json'))
    
    print(f"Processing {len(json_files)} files...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = json_file.name
            
            # Extract task and subtask
            task, subtask = extract_task_subtask_info(data, filename)
            print(f"Task: {task}, Subtask: {subtask}")
            # Process each question
            for question in data.get('questions', []):
                question_type = question.get('question_type', '')
                modality = extract_modality_from_question_type(question_type)
                
                all_data.append({
                    'task': task,
                    'subtask': subtask,
                    'modality': modality,
                    'is_correct': question.get('is_correct', False)
                })
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    df = pd.DataFrame(all_data)
    
    if len(df) == 0:
        print("No data found!")
        return
    
    # Calculate average accuracy for each modality-task-subtask combination
    summary = df.groupby(['modality', 'task', 'subtask']).agg({
        'is_correct': ['count', 'mean']
    }).round(4)
    
    summary.columns = ['count', 'accuracy']
    summary['accuracy_pct'] = (summary['accuracy'] * 100).round(2)
    summary['count'] = summary['count'] / 6
    summary = summary.reset_index()
    
    # Create task-subtask combinations
    summary['task_subtask'] = summary['task'] + '_' + summary['subtask']
        
    # Create pivot table
    accuracy_pivot = summary.pivot_table(
        index='modality',
        columns='task_subtask', 
        values='accuracy_pct',
        fill_value=0
    )
    
    # Sort modalities in the desired order
    modality_order = [
        'Audio -> Text',
        'Audio -> Vision', 
        'Text -> Audio',
        'Text - Vision',
        'Vision -> Audio',
        'Vision -> Text'
    ]
    
    # Reorder rows
    available_modalities = [m for m in modality_order if m in accuracy_pivot.index]
    accuracy_pivot = accuracy_pivot.reindex(available_modalities)
    
    # Sort columns
    columns = list(accuracy_pivot.columns)
    columns.sort()
    accuracy_pivot = accuracy_pivot[columns]
    
    # Create the CSV in the exact format shown
    # First, create the header rows
    output_data = []
    
    # Statistics row (counts)
    stats_row = ['Statistics (all_count)', ''] + [str(int(summary[summary['task_subtask'] == col]['count'].sum())) if col in summary['task_subtask'].values else '0' for col in accuracy_pivot.columns]
    
    # Model row (empty spaces represented by commas)
    model_row = ['Model', ''] + [','.join([''] * len(col.split('_'))) for col in accuracy_pivot.columns]
    
    # Task header row
    task_headers = []
    current_task = None
    for col in accuracy_pivot.columns:
        task_part = col.split('_')[0] + '_' + col.split('_')[1] if '_' in col else col
        if task_part != current_task:
            current_task = task_part
            if 'perception' in col.lower():
                task_headers.append('T1 (Perception)')
            elif 'spatial' in col.lower():
                task_headers.append('T2 Spatial')
            elif 'temporal' in col.lower():
                task_headers.append('T3 Temporal')
            elif 'speech' in col.lower():
                task_headers.append('T4 Speech')
            elif 'external' in col.lower():
                task_headers.append('T5 External')
            else:
                task_headers.append(task_part)
        else:
            task_headers.append('')
    
    task_row = ['', ''] + task_headers
    
    # Define the exact order of columns as specified
    column_order = [
        'General', 'General - Hard', 'Scene', 'Instruments', 'Instruments-multi', 
        'Arrangement', 'Moving Direction', 'Indoor', 'Order', 'Counting', 
        'Calculation', 'Recognition', 'Translation', 'Genre', 'Emotion', 'Movie', 'Singer'
    ]
    
    # Create mapping from actual column names to desired order
    subtask_mapping = {
        'general_activities': 'General',
        'finegrained': 'General - Hard', 
        'natures': 'Scene',
        'instruments': 'Instruments',
        'instruments_comp': 'Instruments-multi',
        'arrangements': 'Arrangement',
        '3D_movements': 'Moving Direction',
        'panaroma': 'Indoor',
        'order': 'Order',
        'count': 'Counting',
        'calculation': 'Calculation',
        'recognition': 'Recognition',
        'translation': 'Translation',
        'music_genre_classification': 'Genre',
        'emotion_classification': 'Emotion',
        'movie_matching': 'Movie',
        'singer_identification': 'Singer'
    }
    
    # Reorder columns based on the specified order
    column_mapping = {}
    for col in accuracy_pivot.columns:
        subtask = '_'.join(col.split('_')[2:])
        mapped_name = subtask_mapping.get(subtask, subtask)
        if mapped_name in column_order:
            column_mapping[mapped_name] = col
    
    # Create new ordered columns list
    ordered_columns = []
    for desired_col in column_order:
        if desired_col in column_mapping:
            ordered_columns.append(column_mapping[desired_col])
    
    # Reorder the pivot table
    accuracy_pivot = accuracy_pivot[ordered_columns]
    
    # Subtask headers in the specified order
    subtask_headers = [subtask_mapping.get('_'.join(col.split('_')[2:]), '_'.join(col.split('_')[2:])) for col in accuracy_pivot.columns]
    
    subtask_row = ['', ''] + subtask_headers
    
    # Add header rows
    output_data.append(stats_row)
    output_data.append(task_row)
    output_data.append(subtask_row)
    
    # Add model name and data rows
    model_name = 'Human'
    for modality in accuracy_pivot.index:
        row = [model_name, modality]
        for col in accuracy_pivot.columns:
            value = accuracy_pivot.loc[modality, col]
            row.append(f"{value:.2f}" if value > 0 else "0.00")
        output_data.append(row)
        model_name = ''  # Only show model name in first row
    
    # Convert to DataFrame and save
    max_cols = max(len(row) for row in output_data)
    for row in output_data:
        while len(row) < max_cols:
            row.append('')
    
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv, index=False, header=False)
    
    print(f"CSV saved to: {output_csv}")
    print(f"Processed {len(df)} questions from {summary['task_subtask'].nunique()} different task-subtask combinations")
    
    return output_df

# Main execution
if __name__ == "__main__":
    input_directory = "/home/xwang378/scratch/2025/AudioBench/user_results"  # Change this to your folder path
    
    try:
        result_df = process_all_user_files_and_create_csv(input_directory, "human_results_table.csv")
        print("CSV file created successfully in the exact format requested!")
        
    except Exception as e:
        print(f"Error: {e}")