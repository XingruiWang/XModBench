import os
from google import genai
from google.genai import types
from tqdm import tqdm
# 替换为你的Gemini API密钥
API_KEY = "AIzaSyBN4640U0aEeDmEwijkcmyFZ-WwnASRThM"


# 初始化Gemini客户端
client = genai.Client(api_key="AIzaSyBN4640U0aEeDmEwijkcmyFZ-WwnASRThM")

def filter_audio_with_gemini(folder_path):
    """
    遍历指定文件夹下的所有.wav文件，并使用Gemini API进行过滤。

    Args:
        folder_path (str): 包含音频文件的文件夹路径。
    """
    print(f"开始处理文件夹: {folder_path}\n")

    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在。")
        return
    
    low_quality_samples = []

    # 遍历文件夹中的所有文件
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".wav"):
            audio_path = os.path.join(folder_path, filename)
            
            try:
                # 读取音频数据
                with open(audio_path, "rb") as audio_file:
                    audio_data = audio_file.read()

                # 设置Gemini的prompt
                prompt = """Listen to this audio carefully and analyze whether there are repeated actions or sounds. 
                    Look for patterns such as:
                    - Repeated movements (footsteps, clapping, tapping, etc.)
                    - Cyclical actions (bouncing, drumming, sawing, etc.)
                    - Any sounds that occur multiple times in a similar pattern
                    - Rhythmic or periodic activities
                    - The audio has no background noise or other human voice
                    """
                    

                # 调用Gemini API
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[
                        prompt,
                        "Below is example of low quality audio.",
                        types.Part.from_bytes(mime_type="audio/wav", data=open("/home/xwang378/scratch/2025/AudioBench/benchmark/Data/ExtremCountAV/-0HwkO7TRmc.00.wav", "rb").read()),

                        "Please respond with only 'Yes' if you can clearly hear repeated actions, or 'No' if you cannot detect any repeated actions.",
                        types.Part.from_bytes(mime_type="audio/wav", data=audio_data)
                    ]
                )
                
                # 解析并打印结果
                result = response.text.strip().upper()
                if result.lower() == "no":
                    low_quality_samples.append(audio_path)
                    print(f"  {filename}: Gemini -> {result}")
                else:
                    print(f"  {filename}: Gemini -> {result}")

            except Exception as e:
                print(f"  处理文件 {filename} 时发生错误: {e}")
    return low_quality_samples

# 设置你的音频文件路径
audio_folder_path = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/ExtremCountAV"
low_quality_samples = filter_audio_with_gemini(audio_folder_path)

print(f"低质量样本数量: {len(low_quality_samples)}")


with open("low_quality_samples.txt", "w") as f:
    for sample in low_quality_samples:
        f.write(sample + "\n")