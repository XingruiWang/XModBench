import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data
tasks = {
    "T1 Perceptual Recognition": {
        "General Categories": 1000,
        "Fine-grained Categories": 1000,
        "Natural Environment": 500,
        "Instruments": 1000,
        "Instrument Composition": 500
    },
    "T2 Spatial Reasoning": {
        "2D Horizontal Arrangement": 465,
        "3D Localization": 390,
        "3D Movements": 509
    },
    "T3 Temporal Reasoning": {
        "Event Order": 500,
        "Repetition Count": 411,
        "Repetition Calculation": 439
    },
    "T4 Linguistic Understanding": {
        "Linguistic Recognition": 672,
        "Translation": 702,
        "Dialogue Emotion": 700
    },
    "T5 External Knowledge": {
        "Music Genre Classification": 1000,
        "Movie Matching": 200,
        "Singer Identification": 150
    }
}

# Prepare data
task_labels = list(tasks.keys())
task_sizes = [sum(sub.values()) for sub in tasks.values()]

sub_labels = []
sub_sizes = []
sub_colors = []
task_colors = []

def create_color_palette(base_color, num_colors, lightness_range=(0.3, 0.8)):
    """Create a harmonious color palette based on a base color"""
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    import matplotlib.colors as mcolors
    
    # Convert base color to HSV
    rgb = mcolors.to_rgb(base_color)
    hsv = rgb_to_hsv(rgb)
    
    colors = []
    lightness_values = np.linspace(lightness_range[0], lightness_range[1], num_colors)
    
    for i, lightness in enumerate(lightness_values):
        # Vary saturation slightly for more interest
        sat_variation = 0.1 * np.sin(i * np.pi / num_colors)
        new_hsv = [hsv[0], 
                   max(0.4, min(1.0, hsv[1] + sat_variation)), 
                   lightness]
        rgb = hsv_to_rgb(new_hsv)
        colors.append(rgb)
    
    return colors

# Enhanced color scheme with better contrast
base_colors = ['#2E8B57', '#FF7F50', '#4682B4', '#DA70D6', '#32CD32']  # More distinct colors
task_color_names = ['Forest Green', 'Coral', 'Steel Blue', 'Orchid', 'Lime Green']

for i, (task, subs) in enumerate(tasks.items()):
    values = list(subs.values())
    
    # Create color palette for this task
    colors = create_color_palette(base_colors[i], len(values) + 1)
    
    # Main task color (slightly darker)
    task_colors.append(colors[0])
    
    # Subtask colors (lighter variations)
    for j, (sub, val) in enumerate(subs.items()):
        sub_labels.append(sub)
        sub_sizes.append(val)
        sub_colors.append(colors[j + 1])

# Create the plot with improved styling
plt.style.use('default')  # Use clean default style
fig, ax = plt.subplots(figsize=(14, 10))

# Inner ring (Tasks) - with percentage labels
def autopct_format(pct):
    return f'{pct:.1f}%' if pct > 5 else ''  # Only show percentage if > 5%

wedges1, texts1, autotexts1 = ax.pie(
    task_sizes, 
    radius=0.65, 
    labels=None,
    autopct=autopct_format,
    pctdistance=0.4,
    colors=task_colors,
    textprops={'fontsize': 10, 'weight': 'bold', 'color': 'white'},
    wedgeprops=dict(width=0.35, edgecolor='white', linewidth=2)
)

# Outer ring (Subtasks)
wedges2, texts2 = ax.pie(
    sub_sizes, 
    radius=1.0, 
    labels=None,
    colors=sub_colors,
    wedgeprops=dict(width=0.25, edgecolor='white', linewidth=1.5)
)

# Add center circle for cleaner look
centre_circle = plt.Circle((0, 0), 0.3, fc='white', linewidth=2, edgecolor='#cccccc')
ax.add_artist(centre_circle)

# Create custom legend
legend_elements = []
for i, (task, color) in enumerate(zip(task_labels, task_colors)):
    # Clean up task names for legend
    clean_name = task.replace('T' + str(i+1) + ' ', '')
    legend_elements.append(mpatches.Patch(color=color, label=clean_name))

# Position legend to the right
ax.legend(handles=legend_elements, 
         loc='center left', 
         bbox_to_anchor=(1.1, 0.5),
         fontsize=11,
         frameon=True,
         fancybox=True,
         shadow=True)

# Add title and subtitle
plt.suptitle('Audio Benchmark Task Distribution', 
            fontsize=18, 
            weight='bold', 
            y=0.95)

plt.figtext(0.5, 0.90, 
           'Inner ring: Main tasks | Outer ring: Subtasks', 
           ha='center', 
           fontsize=12, 
           style='italic',
           color='#666666')

# Add total sample count
total_samples = sum(task_sizes)
plt.figtext(0.5, 0.02, 
           f'Total samples: {total_samples:,}', 
           ha='center', 
           fontsize=10, 
           weight='bold',
           color='#333333')

ax.set_aspect("equal")

# Improve layout
plt.tight_layout()

# Save with high quality
plt.savefig('improved_benchmark_distribution.png', 
           dpi=300, 
           bbox_inches='tight',
           facecolor='white',
           edgecolor='none')

plt.show()

# Optional: Create a detailed breakdown table
print("\nDetailed Task Breakdown:")
print("=" * 50)
for task, subtasks in tasks.items():
    task_total = sum(subtasks.values())
    print(f"\n{task}: {task_total:,} samples")
    for subtask, count in subtasks.items():
        percentage = (count / task_total) * 100
        print(f"  • {subtask}: {count:,} ({percentage:.1f}%)")