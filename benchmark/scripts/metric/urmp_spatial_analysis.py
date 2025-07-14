import json
import os
import matplotlib
matplotlib.use('Agg')

Root = '/home/xwang378/scratch/2025/AudioBench/benchmark/results/'
model_name = "gemini-2.0-flash"

preload_task = '/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/02_spatial/urmp/urmp_audio_bench_questions_audio_text.json'

with open(preload_task, 'r') as f:
    preload_task = json.load(f)


result_dict = {}
for task_name in ['spatial_urmp_audio_text', 
                  'spatial_urmp_text_audio', 
                  'spatial_urmp_audio_vision', 
                  'spatial_urmp_vision_audio', 
                  'spatial_urmp_vision_text', 
                  'spatial_urmp_text_vision']:
    result_dict[task_name] = {}
    file_path = os.path.join(Root, model_name, task_name + '.json')
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    for i, result_idx in enumerate(results):
        result = results[result_idx]
        count = preload_task[i]['extra_info']['count']
        if count not in result_dict[task_name]:
            result_dict[task_name][count] = {
                'correct_count': 0,
                'all_count': 0,
            }

        if bool(result['is_correct']):
            result_dict[task_name][count]['correct_count'] += 1
        result_dict[task_name][count]['all_count'] += 1

for task_name in result_dict:
    for count in result_dict[task_name]:
        print(f"{task_name} {count} {result_dict[task_name][count]['correct_count'] / result_dict[task_name][count]['all_count'] * 100:.2f}%")



import matplotlib.pyplot as plt
import numpy as np

# Prepare data for grouped bar plot
task_names = list(result_dict.keys())
counts = sorted({count for task in result_dict.values() for count in task.keys()})

# Build a matrix of accuracies: rows=task_names, cols=counts
accuracy_matrix = []
for task_name in task_names:
    row = []
    for count in counts:
        if count in result_dict[task_name]:
            correct = result_dict[task_name][count]['correct_count']
            total = result_dict[task_name][count]['all_count']
            acc = correct / total * 100 if total > 0 else 0
        else:
            acc = 0
        row.append(acc)
    accuracy_matrix.append(row)

# Grouped bar plot
# For each task, sort the counts and plot bars in that order
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 7))

# For each count, then for each task, plot the accuracy
all_counts = sorted({count for task in result_dict.values() for count in task.keys()})
x = np.arange(len(all_counts))

for i, task_name in enumerate(task_names):
    accs = []
    for count in all_counts:
        if count in result_dict[task_name]:
            correct = result_dict[task_name][count]['correct_count']
            total = result_dict[task_name][count]['all_count']
            acc = correct / total * 100 if total > 0 else 0
        else:
            acc = 0
        accs.append(acc)
    ax.bar(x + i * width, accs, width, label=task_name)

# Set x-ticks and labels
ax.set_xticks(x + width * (len(task_names) - 1) / 2)
ax.set_xticklabels([str(c) for c in all_counts])

ax.set_xlabel('Number of Instruments')
ax.set_ylabel('Accuracy (%)')
ax.set_title('URMP Spatial Task Accuracy by Group Size')
ax.legend(title='Task Name')
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(f'urmp_spatial_analysis.png')
