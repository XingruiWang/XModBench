import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Set style for better aesthetics
plt.style.use('default')
sns.set_palette("husl")

# -------- Data ----------
data = {
    "Model": [
        "PandaGPT","Unified-IO 2 XXL","Unified-IO 2 XL","Unified-IO 2",
        "VITA","VideoLLaMA 2","Baichuan Omni 1.5","EchoInk-R1","Qwen2.5-Omni",
        "Gemini 1.5 Pro","Gemini 2.0 Flash","Gemini 2.5 Flash","Gemini 2.5 Pro"
    ],
    "Vision-Text": [1.13,-16.99,-6.91,-4.65,-14.10,-23.02,-13.89,-25.18,-18.91,-22.09,-21.36,-14.58,-15.63],
    "Audio-Vision": [-2.01,-9.01,-9.84,-10.35,-16.05,-19.00,-40.92,-17.52,-18.51,-49.37,-40.74,-44.01,-33.01],
    "Audio-Text": [-0.88,-26.01,-16.75,-15.00,-30.15,-42.02,-54.82,-42.70,-37.41,-71.46,-62.10,-58.59,-48.63]
}
df = pd.DataFrame(data)

# -------- Enhanced Plot ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Improved color palette - more sophisticated and distinct
colors = {
    "Vision-Text": "#2E86AB",    # Professional blue
    "Audio-Vision": "#F18F01",   # Warm orange
    "Audio-Text": "#A23B72"      # Deep magenta
}

# Add subtle background color
fig.patch.set_facecolor('#FAFAFA')


# -------- Alternative: Single plot with grouped bars ----------
def create_grouped_barplot():
    """Alternative visualization with grouped bars"""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Prepare data for grouped bars
    df_melted = df.melt(id_vars=['Model'], var_name='Modality', value_name='Value')
    # Keep original negative values - don't convert to positive
    
    # Create grouped bar plot
    x = np.arange(len(df['Model']))
    width = 0.25
    
    modalities = ['Vision-Text', 'Audio-Vision', 'Audio-Text']
    modalities_names = ['Between Vision & Text', 'Between Audio & Vision', 'Between Audio & Text']
    
    for i, modality in enumerate(modalities):
        values = df[modality]  # Keep original negative values
        bars = ax.bar(x + i*width, values, width, 
                     label=modalities_names[i], 
                     color=colors[modality], alpha=0.8,
                     edgecolor='white', linewidth=1)
    
    # Styling
    ax.set_xlabel('Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Difference (closer to 0 is better)', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Comparison - All Modalities', 
                fontsize=18, fontweight='bold', pad=20)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=18)
    # y tick labels fontsize 18
    ax.tick_params(axis='y', labelsize=18)
    
    # Invert y-axis so negative values go down
    ax.invert_yaxis()
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=18)
    # hide legend
    
    ax.legend().set_visible(False)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("grouped_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

# Uncomment to create the alternative grouped visualization
create_grouped_barplot()