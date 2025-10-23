import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -------- Data ----------
data = {
    "Model": [
        "Panda","Unified Io 2 Xxl","Unified Io 2 Xl","Unified Io 2",
        "Vita","VideoLLaMA 2","Baichuan Omni 1.5","Echoink","Qwen2.5 Omni",
        "Gemini 1.5 Pro","Gemini 2.0 Flash","Gemini 2.5 Flash","Gemini 2.5 Pro"
    ],
    "Vision-Text": [1.13,-16.99,-6.91,-4.65,-14.10,-23.02,-13.89,-25.18,-18.91,-22.09,-21.36,-14.58,-15.63],
    "Audio-Text": [-0.88,-26.01,-16.75,-15.00,-30.15,-42.02,-54.82,-42.70,-37.41,-71.46,-62.10,-58.59,-48.63],
    "Audio-Vision": [-2.01,-9.01,-9.84,-10.35,-16.05,-19.00,-40.92,-17.52,-18.51,-49.37,-40.74,-44.01,-33.01]
}
df = pd.DataFrame(data)

# -------- Plot ----------
fig, ax = plt.subplots(figsize=(12, 8))

y = np.arange(len(df))
bar_height = 0.2  # 每组的厚度
gap = 0.05        # 组内柱子之间的间距

# 三个并列的水平柱
ax.barh(y - bar_height - gap, -df["Vision-Text"], height=bar_height, color="#AFFC41", label="Capability comparison: Vision - Text")
ax.barh(y + bar_height + gap, -df["Audio-Text"], height=bar_height, color="#086375", label="Capability comparison: Audio - Text")
ax.barh(y, -df["Audio-Vision"], height=bar_height, color="#1DD3B0", label="Capability comparison: Audio - Vision")

# 设置 y 轴
ax.set_yticks(y)
ax.set_yticklabels(df["Model"], fontsize=20)

# x 轴范围
ax.set_xlim(-5, 100)
ax.set_xticks(np.arange(-5, 100, 20))
ax.set_xticklabels(np.arange(5, -100, -20), fontsize=18)

# add line

ax.set_xlabel("Difference Value")
# ax.set_title("Parallel Horizontal Bar Chart (Three Modalities)")
ax.legend(fontsize=10)

plt.tight_layout()
plt.style.use('ggplot')  # ggplot 风格
ax.set_facecolor("white")  # 白色背景
# plt.show()
plt.savefig("relative.png")

