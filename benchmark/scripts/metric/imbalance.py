import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -------- Data ----------
data = {
    "Model": [
        "Panda","Unified Io 2 Xxl","Unified Io 2 Xl","Unified Io 2",
        "Vita","VideoLLama 2","Baichuan Omni 1.5","Echoink","Qwen2.5 Omni",
        "Gemini 1.5 Pro","Gemini 2.0 Flash","Gemini 2.5 Flash","Gemini 2.5 Pro"
    ],
    "A_vs_T_imb": [0.77,6.24,6.22,3.47,10.42,22.85,7.36,8.16,6.67,3.74,11.48,7.55,6.60],
    "A_vs_T_avg": [24.16,34.32,30.23,27.18,35.03,37.13,44.15,60.49,58.69,50.48,57.97,58.84,67.69],
    "V_vs_T_imb": [-0.08,2.06,4.47,0.66,32.55,40.28,16.73,16.64,16.62,9.50,13.72,10.26,8.82],
    "V_vs_T_avg": [25.16,38.82,35.15,32.36,43.06,46.62,64.61,69.25,67.95,75.17,78.34,80.84,84.19],
    "V_vs_A_imb": [-0.48,1.69,-0.48,1.70,3.90,-0.82,2.79,3.98,2.53,2.43,0.67,0.74,1.91],
    "V_vs_A_avg": [24.72,25.82,26.77,24.86,27.98,25.61,37.20,47.90,49.24,39.44,47.29,51.55,59.88]
}
df = pd.DataFrame(data)

# -------- Plot ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

comparisons = [
    ("A vs T", "A_vs_T_avg", "A_vs_T_imb"),
    ("V vs T", "V_vs_T_avg", "V_vs_T_imb"),
    ("V vs A", "V_vs_A_avg", "V_vs_A_imb"),
]

for ax, (title, avg_col, imb_col) in zip(axes, comparisons):
    x = np.arange(len(df))
    # barplot for average
    ax.bar(x, df[avg_col], color="skyblue", alpha=0.7, label="Average")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=90)
    ax.set_ylabel("Average")
    ax.set_title(title)
    
    # secondary axis for imbalance
    ax2 = ax.twinx()
    ax2.plot(x, df[imb_col], color="darkred", marker="o", linewidth=2, label="Imbalance")
    ax2.set_ylabel("Imbalance")
    
    # legends
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

plt.tight_layout()
# plt.show()

plt.savefig("imbalance3.png", dpi=300, bbox_inches='tight')  
