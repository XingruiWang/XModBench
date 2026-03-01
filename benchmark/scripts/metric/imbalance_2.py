import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set style for better aesthetics
plt.style.use('default')

# -------- Three datasets ----------
models = [
    "PandaGPT","Unified-IO 2 XXL","Unified-IO 2 XL","Unified-IO 2",
    "VITA","VideoLLaMA 2","Baichuan Omni 1.5","EchoInk-R1","Qwen2.5-Omni",
    "Gemini 1.5 Pro","Gemini 2.0 Flash","Gemini 2.5 Flash","Gemini 2.5 Pro"
]

# Vision ↔ Text data
vision_text_data = {
    "Model": models,
    "X_to_Y": [25.12,39.85,37.39,32.69,59.33,66.76,72.98,77.57,76.26,79.92,85.20,85.98,88.60],
    "Y_to_X": [25.20,37.79,32.91,32.03,26.78,26.48,56.25,60.93,59.64,70.42,71.48,75.71,79.78],
    "imbalance": [-0.08,2.06,4.47,0.66,32.55,40.28,16.73,16.64,16.62,9.50,13.72,10.26,8.82]
}

# Audio ↔ Text data
audio_text_data = {
    "Model": models,
    "X_to_Y": [24.54,37.43,33.34,28.92,40.24,48.55,47.83,64.57,62.03,52.36,63.71,62.61,67.69],
    "Y_to_X": [23.77,31.20,27.12,25.45,29.82,25.70,40.47,56.41,55.36,48.61,52.23,55.06,61.09],
    "imbalance": [0.77,6.24,6.22,3.47,10.42,22.85,7.36,8.16,6.67,3.74,11.48,7.55,6.60]
}

# Audio ↔ Vision data
audio_vision_data = {
    "Model": models,
    "X_to_Y": [24.48,26.67,26.53,25.71,29.93,25.21,38.60,49.89,50.50,40.65,47.63,51.92,60.83],
    "Y_to_X": [24.96,24.97,27.01,24.01,26.03,26.02,35.81,45.91,47.98,38.22,46.96,51.18,58.92],
    "imbalance": [-0.48,1.69,-0.48,1.70,3.90,-0.82,2.79,3.98,2.53,2.43,0.67,0.74,1.91]
}

# Create DataFrames
df_vt = pd.DataFrame(vision_text_data)
df_at = pd.DataFrame(audio_text_data)
df_av = pd.DataFrame(audio_vision_data)

def create_subplot(ax1, df, title, x_label1, x_label2, colors):
    """Function to create individual subplot"""
    
    y = np.arange(len(df))
    bar_width = 0.6
    
    # Create main horizontal bar charts
    bars1 = ax1.barh(y, df["X_to_Y"], height=bar_width, 
                     color=colors[0], alpha=0.8, 
                     label=x_label1, edgecolor='white', linewidth=1)
    
    bars2 = ax1.barh(y, -df["Y_to_X"], height=bar_width, 
                     color=colors[1], alpha=0.8, 
                     label=x_label2, edgecolor='white', linewidth=1)
    
    # Set main axis
    ax1.set_yticks(y)
    ax1.set_yticklabels(df["Model"], fontsize=15, fontweight='bold')
    ax1.set_xlabel("Performance Score", fontsize=12, fontweight='bold', color='#2C3E50')
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=15, color='#2C3E50')
    ax1.set_xlim(-100, 100)
    
    # Add zero line
    ax1.axvline(0, color='#34495E', linewidth=2, alpha=0.8)
    
    # Grid lines
    ax1.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Create secondary axis for imbalance
    ax2 = ax1.twiny()
    bars3 = ax2.barh(y, df["imbalance"], height=bar_width*0.4, 
                     color=colors[2], alpha=0.9, 
                     label="Imbalance", edgecolor='white', linewidth=1)
    
    ax2.set_xlabel("Imbalance", fontsize=20, fontweight='bold', color='#2C3E50')
    
    # Adjust imbalance axis based on data range
    imbalance_max = max(abs(df["imbalance"].min()), abs(df["imbalance"].max()))
    # ax2.set_xlim(-imbalance_max*1.2, imbalance_max*1.2)
    ax2.set_xlim(-50, 50)
    for i, bar in enumerate(bars3):
        width = bar.get_width()
        if width >= 0:
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
                     f'{width:.1f}', va='center', ha='left',
                     fontsize=12, fontweight='bold', color=colors[2])
        else:
            ax2.text(width - 1, bar.get_y() + bar.get_height()/2,
                     f'{width:.1f}', va='center', ha='right',
                     fontsize=12, fontweight='bold', color=colors[2])
    # Add value labels (only on first subplot to avoid crowding)
    # if title == "Vision ↔ Text":
    #     for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    #         # Right side values
    #         ax1.text(bar1.get_width() + 2, bar1.get_y() + bar1.get_height()/2, 
    #                  f'{df["X_to_Y"].iloc[i]:.0f}', 
    #                  va='center', ha='left', fontsize=12, fontweight='bold', color='#2C3E50')
    #         # Left side values
    #         ax1.text(bar2.get_width() - 2, bar2.get_y() + bar2.get_height()/2, 
    #                  f'{df["Y_to_X"].iloc[i]:.0f}', 
    #                  va='center', ha='right', fontsize=12, fontweight='bold', color='#2C3E50')
    
    # Style the axes
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#7F8C8D')
    ax1.spines['bottom'].set_color('#7F8C8D')
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    
    ax2.spines['top'].set_color('#7F8C8D')
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_linewidth(1.5)
    
    # Tick styling
    ax1.tick_params(axis='x', labelsize=10, colors='#2C3E50')
    ax1.tick_params(axis='y', labelsize=10, colors='#2C3E50')
    ax2.tick_params(axis='x', labelsize=10, colors='#2C3E50')
    
    # Add background regions
    ax1.axvspan(-100, 0, alpha=0.05, color=colors[1])
    ax1.axvspan(0, 100, alpha=0.05, color=colors[0])
    
    return ax1, ax2

# -------- Create three subplots ----------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Remove background
fig.patch.set_facecolor('white')

# Define harmonious color schemes for each subplot
# Blue family - cool, professional
colors_vt = ["#5DADE2", "#85C1E9", "#E74C3C"]  # Light blue, lighter blue, red accent

# Green family - natural, balanced  
colors_at = ["#58D68D", "#82E0AA", "#E74C3C"]  # Medium green, light green, orange accent

# Purple family - creative, sophisticated
colors_av = ["#BB8FCE", "#D7BDE2", "#E74C3C"]  # Medium purple, light purple, yellow accent

# Create three subplots
ax1_main, ax1_sec = create_subplot(ax1, df_vt, "Vision ↔ Text", 
                                   "Vision → Text", "Text → Vision", colors_vt)
ax2_main, ax2_sec = create_subplot(ax2, df_at, "Audio ↔ Text", 
                                   "Audio → Text", "Text → Audio", colors_at)
ax3_main, ax3_sec = create_subplot(ax3, df_av, "Vision ↔ Audio", 
                                   "Audio → Vision", "Vision → Audio", colors_av)

# Hide y-axis labels on second and third subplots to save space
ax2.set_yticklabels([])
ax3.set_yticklabels([])

# Add legends for each subplot (combine main and secondary handles)
handles1_main, labels1_main = ax1_main.get_legend_handles_labels()
handles1_sec, labels1_sec = ax1_sec.get_legend_handles_labels()
legend1 = ax1_main.legend(handles=handles1_main + handles1_sec, loc="lower right", frameon=True, fancybox=True, shadow=True, 
                        fontsize=10, framealpha=0.9)
legend1.get_frame().set_facecolor('white')
legend1.get_frame().set_edgecolor('#BDC3C7')

handles2_main, labels2_main = ax2_main.get_legend_handles_labels()
handles2_sec, labels2_sec = ax2_sec.get_legend_handles_labels()
legend2 = ax2_main.legend(handles=handles2_main + handles2_sec, loc="lower right", frameon=True, fancybox=True, shadow=True, 
                        fontsize=10, framealpha=0.9)
legend2.get_frame().set_facecolor('white')
legend2.get_frame().set_edgecolor('#BDC3C7')

handles3_main, labels3_main = ax3_main.get_legend_handles_labels()
handles3_sec, labels3_sec = ax3_sec.get_legend_handles_labels()
legend3 = ax3_main.legend(handles=handles3_main + handles3_sec, loc="lower right", frameon=True, fancybox=True, shadow=True, 
                        fontsize=10, framealpha=0.9)
legend3.get_frame().set_facecolor('white')
legend3.get_frame().set_edgecolor('#BDC3C7')

# Adjust subplot spacing
plt.tight_layout()
plt.subplots_adjust(wspace=0.15)

# Add main title
fig.suptitle('Multi-Modal Performance Comparison with Imbalance Analysis', 
             fontsize=18, fontweight='bold', y=0.95, color='#2C3E50')

plt.savefig("three_modal_comparison_harmonious.png", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()