import os

root = "/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/process/5_Exteral/singers"  # 你的 singers 文件夹路径

no_audio = []
no_image = []

for singer in os.listdir(root):
    singer_path = os.path.join(root, singer)
    if not os.path.isdir(singer_path):
        continue
    
    # 检查 audio
    audio_path = os.path.join(singer_path, "audio")
    has_audio = os.path.isdir(audio_path) and len(os.listdir(audio_path)) > 0

    # 检查 image
    images_path = os.path.join(singer_path, "images")
    has_images = (
        (os.path.isdir(images_path) and len(os.listdir(images_path)) > 0)
        or os.path.exists(os.path.join(singer_path, "images.csv"))
    )

    if not has_audio:
        no_audio.append(singer)
    if not has_images:
        no_image.append(singer)

print("没有 audio 的 singers:")
print(no_audio)

print("\n没有 image 的 singers:")
print(no_image)