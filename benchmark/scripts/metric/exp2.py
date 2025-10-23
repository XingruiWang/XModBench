import pandas as pd
import numpy as np
import os
import seaborn as sns
# -------- Step 1: Load the table data ----------
# Ideally copy the LaTeX tables into CSV or Excel manually
# Example schema for each task:
# columns = ["Model", "A->T", "A->V", "T->A", "T->V", "V->A", "V->T", "Avg"]

# Suppose you already exported each task table to CSV:
task1 = pd.read_csv("task1.csv")
task2 = pd.read_csv("task2.csv")
task3 = pd.read_csv("task3.csv")
task4 = pd.read_csv("task4.csv")
task5 = pd.read_csv("task5.csv")

tasks = {
    "Task1": task1,
    "Task2": task2,
    "Task3": task3,
    "Task4": task4,
    "Task5": task5
}

# -------- Step 2: Define metrics ----------
def compute_scores(df):
    results = []
    for _, row in df.iterrows():
        model = row["Model"]

        # Handle missing values ("-" or NaN)
        def val(x):
            try:
                return float(row[x])
            except:
                return np.nan

        a_t, v_t = val("Audio -> Text"), val("Vision -> Text")
        t_a, t_v = val("Text -> Audio"), val("Text -> Vision")
        a_v, v_a = val("Audio -> Vision"), val("Vision -> Audio")

        # 1. Base modality score (average of A->T and V->T)
        In_Audio = a_t
        In_Vision = v_t

        # 2. Output-Modality Scores
        out_audio  = np.nanmean([t_a, v_a])   # Text->Audio, Vision->Audio
        out_vision = np.nanmean([t_v, a_v])   # Text->Vision, Audio->Vision
        out_text   = np.nanmean([a_t, v_t])   # Audio->Text, Vision->Text

        results.append({
            "Model": model,
            "In_Audio": In_Audio,
            "In_Vision": In_Vision,
            "Out_Audio": out_audio / out_text * 100,
            "Out_Vision": out_vision / out_text * 100,
            # "Out_Vision": out_vision ,
            "Out_Text": out_text
        })
    return pd.DataFrame(results)

# -------- Step 3: Apply to all tasks ----------
all_results = {}
for tname, df in tasks.items():
    all_results[tname] = compute_scores(df)

# -------- Step 4: Concatenate for global view ----------
summary = pd.concat(
    {task: df.set_index("Model") for task, df in all_results.items()},
    axis=0
)
# import ipdb; ipdb.set_trace()
# summary.columns = ["Task","Model","In_Audio","In_Vision","Out_Audio","Out_Vision","Out_Text"]

# -------- Step 5: Save or display ----------
os.makedirs("tables", exist_ok=True)
summary.to_csv("tables/xmodbench_taskwise_scores.csv")
print(summary.head(20))


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读入数据
df = pd.read_csv("tables/xmodbench_taskwise_scores.csv")

# 如果有缺失值先处理一下
df = df.fillna(0)
columns = list(df.columns)
columns[0] = "Task"
df.columns = columns
all_models = df["Model"].unique()

# 归一化到 0-100
# for col in ["In_Audio","In_Vision","Out_Audio","Out_Vision","Out_Text"]:
#     df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) * 100

# 转换成 MultiIndex: (Task, Model)
df.set_index(["Task","Model"], inplace=True)

def plot_radar_per_modality(df, modality, models, tasks):
    # === Step 1: 计算 Overall 平均 ===
    df_avg = df.groupby("Model")[modality].mean()
    
    # 新的维度：Overall + tasks
    tasks_with_overall = ["Overall", "Perception", "Spatial", "Temporal", "Linguistic", "Knowledge"]

    # 角度设置，确保 "Overall" 在顶部 (12点钟方向)
    angles = np.linspace(0, 2*np.pi, len(tasks_with_overall), endpoint=False).tolist()
    angles = [angles[-1]] + angles[:-1]   # rotate so first = top
    angles += angles[:1]
    angles = [(-angle+np.pi/180*30)% (2*np.pi) for angle in angles]

    # 调色盘
    colors = sns.color_palette("Set2", len(models))

    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))

    for i, model in enumerate(models):
        values = []
        # Overall
        try:
            values.append(df_avg.loc[model])
        except KeyError:
            values.append(np.nan)
        # Tasks
        for task in tasks:
            try:
                values.append(df.loc[(task, model), modality])
            except KeyError:
                values.append(np.nan)
        values += values[:1]

        ax.plot(angles, values, label=model, linewidth=2.0, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.15)

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks_with_overall, fontsize=12, fontweight="bold")

    # 设置环形刻度
    ax.set_yticks([20,40,60,80,100])
    ax.set_yticklabels(["20","40","60","80","100"], fontsize=10, color="gray")
    ax.set_ylim(0,100)

    # 标题
    # ax.set_title(f"{modality} across tasks + overall", size=16, weight="bold", pad=20)

    # 图例放外侧，横向排列
    ax.legend(loc="lower center", bbox_to_anchor=(1.25, 1.1), fontsize=9, frameon=False, ncol=8)
    
    # hide legend
    # ax.legend().set_visible(False)

    plt.tight_layout()
    plt.savefig(f"plot/radar_{modality}_with_overall.png", dpi=300, bbox_inches="tight")
    plt.close()

# 示例：挑几个代表模型
tasks = ["Task1","Task2","Task3","Task4","Task5"] 
models = all_models

for modality in ["In_Audio","In_Vision","Out_Audio","Out_Vision","Out_Text"]:
    plot_radar_per_modality(df, modality, models, tasks)


import seaborn as sns
import matplotlib.pyplot as plt

# 重置索引便于 seaborn 使用
df_plot = df.reset_index()

# 选要画的模态
metrics = ["In_Audio","In_Vision","Out_Audio","Out_Vision","Out_Text"]

df_melted = df_plot.melt(
    id_vars=["Task","Model"], 
    value_vars=metrics,
    var_name="Modality", 
    value_name="Score"
)

# 分组柱状图：每个任务一张图
g = sns.catplot(
    data=df_melted, 
    kind="bar", 
    x="Model", 
    y="Score", 
    hue="Modality",
    col="Task", 
    col_wrap=2,  # 每行两个任务
    height=4, 
    aspect=1.5
)
g.set_xticklabels(rotation=90)
plt.tight_layout()
plt.savefig("plot/grouped_bar_alltasks.png")
plt.close()

# === Step: Bar plots per modality ===

# === Step: Bar plots for Out_Vision and Out_Audio ===

# Average across tasks per model
df_overall = df_plot.groupby("Model")[["Out_Vision","Out_Audio", "Out_Text"]].mean()

# remove certain models 
df_overall = df_overall.drop(index=["Panda", "AnyGPT", "Unified Io 2", "Unified Io 2 Xl", "Unified Io 2 Xxl", "OneLLM"])

all_models = df_overall.index
# Keep consistent colors (same as radar)
colors = sns.color_palette("blend:#d9ed92,#1a759f", len(all_models))
all_models_sorted = df_overall.sort_values(by="Out_Text", ascending=False).index
model_color_map = dict(zip(all_models, colors))

for modality in ["Out_Vision", "Out_Audio", "Out_Text"]:
    df_mod = df_overall[[modality]].sort_values(modality, ascending=False).reset_index()


    plt.figure(figsize=(8,6))
    
    bars = plt.bar(
        df_mod["Model"], 
        df_mod[modality], 
        color=[model_color_map[m] for m in df_mod["Model"]]
    )

    plt.xticks(rotation=45, ha="right", fontsize=20)
    # plt.ylabel("Average Score across Tasks", fontsize=12, weight="bold")
    plt.xlabel("")
    # yticks
    plt.ylim(20, 100)
    plt.yticks(np.arange(20, 100, 20), fontsize=20)
    # # plt.title(f"Overall {modality}", fontsize=14, weight="bold")

    # Annotate values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1, f"{height:.1f}",
                 ha='center', va='bottom', fontsize=18)

    plt.tight_layout()
    plt.savefig(f"plot/bar_overall_{modality}.png", dpi=300, bbox_inches="tight")
    plt.close()
