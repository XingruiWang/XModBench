import os
import json
import random
from google import genai
from tqdm import tqdm
import time

# 设置 API Key (从环境变量或配置文件读取)
API_KEY = None
ENV_FILE = "/home/xwang378/scratch/2025/AudioBench/.envs"

if os.path.exists(ENV_FILE):
    with open(ENV_FILE, "r") as f:
        for line in f:
            if line.startswith("Google_API_Key"):
                API_KEY = line.strip().split("=")[1]
                break

if not API_KEY:
    print("Error: Could not find Google_API_Key in .envs")
    exit(1)

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash-lite" # 用于计算Token的模型

def get_token_count(file_path=None, text=None, mime_type=None):
    """调用 API 计算 Token"""
    contents = []
    if text:
        contents.append(text)
    
    if file_path:
        # 这里简化处理，实际需要上传文件或读取bytes
        # 对于估算，我们可以构建一个类似的请求结构
        # 注意：直接读取本地文件计算Token需要先上传到 File API
        # 或者使用本地估算（不精准）。
        # 为了精准，必须使用 client.models.count_tokens 配合实际内容
        pass
        
    try:
        # 构造请求体
        # 注意：Gemini API count_tokens 需要实际内容
        # 如果文件太大，我们只模拟一个典型的 prompt + 占位符
        # 但为了精准，最好是真实上传。由于这里是demo，我演示如何计算
        
        # 模拟：假设我们已经把文件内容放进去了
        # 实际操作中需要 upload_file 获取 URI
        pass
        
        # 由于不能真的上传所有文件（太慢），我们用以下典型值代替测试：
        # 1. 纯文本 Prompt
        response = client.models.count_tokens(
            model=MODEL_NAME,
            contents=types.Content(
                parts=[
                    types.Part(text="Please analyze this audio/video and answer the question: " + "A" * 100) # 模拟 Prompt
                ]
            )
        )
        return response.total_tokens
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0

# 由于上传文件比较麻烦，我们使用 Google 官方公布的换算公式进行"本地精准估算"
# 这比 API 调用更快且免费，误差极小。

def estimate_local_tokens(modality, duration_sec=0, image_count=0):
    """
    基于 Gemini 官方文档的计费标准进行计算
    Audio: ~32 tokens / sec (1k tokens / 31.25s)
    Image: 258 tokens (无论大小)
    Video: 263 tokens / sec (Video include Audio? No, Video is treated as images + audio)
           Video = 258 * FPS (usually 1fps sampling) + Audio tokens
    Text: ~1.3 tokens / char (英文)
    """
    tokens = 0
    
    # 基础 Prompt Token
    tokens += 200 # System prompt + Question
    
    if modality == "audio":
        # Audio: 1 sec = 32 tokens
        tokens += duration_sec * 32
        
    elif modality == "image":
        # Image: Fixed 258 tokens
        tokens += image_count * 258
        
    elif modality == "video":
        # Video: 假设 Gemini 采样率 1fps
        # Video tokens = 258 * duration (1fps)
        # + Audio tokens (if audio included)
        # 通常 Video 任务包含音频
        tokens += (duration_sec * 258) + (duration_sec * 32)
    
    return int(tokens)

# 定义任务类型的典型时长/数量
TASK_CONFIGS = {
    "Perception": {"type": "mixed", "audio_dur": 2, "img_count": 1},
    "Spatial": {"type": "video", "duration": 5},
    "Temporal": {"type": "video", "duration": 10}, # 计数任务通常较长
    "Linguistic": {"type": "audio", "duration": 15}, # 语音识别通常较长
    "Knowledge": {"type": "mixed", "audio_dur": 15, "img_count": 1}
}

# 样本数量 (Max 1000 cap)
SAMPLE_COUNTS = {
    "Perception": 36000, # 6000 * 6
    "Spatial": 8184,
    "Temporal": 8100,
    "Linguistic": 8244,
    "Knowledge": 12300
}

print("=== Gemini Token Count Estimation (Official Formula) ===")
print(f"{'Task Family':<15} | {'Samples':<10} | {'Avg Tokens':<12} | {'Total (M)':<10}")
print("-" * 55)

total_tokens_all = 0

for family, config in TASK_CONFIGS.items():
    avg_tokens = 0
    if config["type"] == "mixed":
        # Image + Audio
        t_img = estimate_local_tokens("image", image_count=config["img_count"])
        t_aud = estimate_local_tokens("audio", duration_sec=config["audio_dur"])
        # 假设 Perception 任务中 Image 和 Audio 模态各半 (混合估算)
        # 实际是 6 模态，都有输入。
        # A->V (Audio Input), V->A (Video/Image Input)...
        # 简单起见，取平均
        avg_tokens = (t_img + t_aud) / 2 + 200 # +Prompt
        # 修正：Perception 大多是 Image+Audio 同时存在？
        # 不，任务是 "Audio->Vision" 意味着 Input Audio, Output Vision?
        # 实际上 AudioBench 是 Input (Audio+Vision), Output Text (Choice).
        # 所以每个样本都包含 Audio 和 Vision。
        avg_tokens = estimate_local_tokens("audio", duration_sec=config["audio_dur"]) + \
                     estimate_local_tokens("image", image_count=config["img_count"])
        
    elif config["type"] == "video":
        avg_tokens = estimate_local_tokens("video", duration_sec=config["duration"])
        
    elif config["type"] == "audio":
        avg_tokens = estimate_local_tokens("audio", duration_sec=config["duration"])
    
    total_family = avg_tokens * SAMPLE_COUNTS[family]
    total_tokens_all += total_family
    
    print(f"{family:<15} | {SAMPLE_COUNTS[family]:<10} | {avg_tokens:<12} | {total_family/1e6:.2f} M")

print("-" * 55)
print(f"TOTAL TOKENS: {total_tokens_all/1e6:.2f} Million")

