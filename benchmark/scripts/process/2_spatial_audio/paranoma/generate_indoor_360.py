import os
import json
import random
import glob
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import subprocess
import tempfile

# ==================== Global Config ====================

# Absolute path root (用于导出题目里 input 字段的绝对路径拼接)
PATH_ROOT = "/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/process/2_spatial_audio"


# ==================== IO / Utils ====================

def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """Load metadata from JSON file"""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def make_absolute_path(relative_path: str) -> str:
    """Convert relative path to absolute path"""
    return os.path.join(PATH_ROOT, relative_path)


def get_azimuth_description(azimuth: int) -> str:
    """Convert azimuth angle (counterclockwise positive, 0° = front) to natural text description
       Angle normalized to [-180, 180).
    """
    # 归一化到 [-180, 180)
    azimuth = (azimuth + 180) % 360 - 180

    if 0 < azimuth < 45:
        return f"slightly to the left of front ({azimuth}°)"
    elif 45 <= azimuth < 90:
        return f"toward the front-left ({azimuth}°)"
    elif 90 <= azimuth < 135:
        return f"toward the back-left ({azimuth}°)"
    elif 135 <= azimuth <= 180:
        return f"slightly to the left of behind ({azimuth}°)"
    elif -45 < azimuth < 0:
        return f"slightly to the right of front ({azimuth}°)"
    elif -90 < azimuth <= -45:
        return f"toward the front-right ({azimuth}°)"
    elif -135 < azimuth <= -90:
        return f"toward the back-right ({azimuth}°)"
    elif -180 <= azimuth <= -135:
        return f"slightly to the right of behind ({azimuth}°)"
    else:
        # 包括 0, -45, -90, -135, 180 等边界
        return f"at {azimuth}°"



# ==================== BBox Rendering Helpers ====================

def azimuth_to_video_position(azimuth: float, elevation: float,
                              video_width: int, video_height: int) -> Tuple[int, int, int, int]:
    """
    Map (azimuth, elevation) to bbox in an equirectangular 360° frame.
    CCW positive azimuth, 0° = front; elevation in [-90, 90] (up positive).
    Returns (x1, y1, x2, y2).
    """
    W, H = video_width, video_height

    # 中心点 X: azimuth -> 水平方向
    center_x = int((0.5 - azimuth / 360.0) * W) % W

    # 中心点 Y: elevation -> 垂直方向 (+90=顶, 0=中, -90=底)
    el = max(-90.0, min(90.0, elevation))
    center_y = int((0.5 - el / 180.0) * H)

    # 框大小（可以调大一些保证覆盖）
    bbox_width = W // 6
    bbox_height = H // 3

    x1 = max(0, center_x - bbox_width // 2)
    x2 = min(W - 1, center_x + bbox_width // 2)
    y1 = max(0, center_y - bbox_height // 2)
    y2 = min(H - 1, center_y + bbox_height // 2)

    return (x1, y1, x2, y2)

def draw_direction_bbox(frame: np.ndarray, azimuth: int, elevation: int, event_class: str) -> np.ndarray:
    """Draw a simple red bounding box on the video frame"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = azimuth_to_video_position(azimuth, elevation, w, h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red box
    cv2.putText(frame, event_class, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame


def add_bbox_to_video(input_video_path: str, output_video_path: str, azimuth: int, elevation: int, event_class: str):
    """
    Add red bounding box to a video file.
    Internally uses OpenCV to write mp4v, then re-encodes with ffmpeg to H.264.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 临时 mp4v
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_with_bbox = draw_direction_bbox(frame, azimuth, elevation, event_class)
        out.write(frame_with_bbox)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[bbox] Processing frame {frame_count}/{total_frames}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Temporary mp4v video saved: {tmp_path}")

    # 统一转码为 H.264
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", tmp_path,
        "-c:v", "libx264", "-crf", "23", "-preset", "medium",
        "-an",
        output_video_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    
    print(f"[INFO] Final H.264 video saved to: {output_video_path}")
    
    os.remove(tmp_path)
    print(f"[INFO] Final H.264 video saved to: {output_video_path}")
    


# ==================== Six Task Generators ====================
# 约定：
# - audio_choice 目录的实例：输入是 Video，选项是 Audio（Video→Audio）等
# - video_choice 目录的实例：输入是 Audio，选项是 Video（Audio→Video）等
# - Text 相关任务：文本描述以事件类别 + 方位自然语言为主，正确项由 metadata['correct_answer'] 决定
# - 所有出现 Video 的输入或选项，均生成 bbox 版本以保持一致性

# ---- 1) Audio -> Video (bbox) ----
def generate_question_audio_video_with_bbox(selected_instance: Dict) -> Dict[str, Any]:
    """Generate audio to video question with bbox-enhanced video choices"""
    metadata_path = os.path.join(selected_instance['path'], selected_instance['metadata_file'])
    metadata = load_metadata(metadata_path)

    input_audio_path = os.path.join(selected_instance['path'], f"{selected_instance['question_id']}_input_audio.wav")
    event_class = metadata['event_info']['class']
    azimuth = metadata['event_info']['azimuth']
    elevation = metadata['event_info']['elevation']
    choice_video_paths = []
    for i in range(4):
        original_video_path = os.path.join(selected_instance['path'], f"{selected_instance['question_id']}_choice_{i}.mp4")
        bbox_video_path = os.path.join(selected_instance['path'], f"{selected_instance['question_id']}_choice_{i}.mp4")

        # 该视角下，声音应该出现在的方位（用来画框）
        choice_rotation = metadata['choice_video_rotations'][i]
        choice_azimuth = (azimuth + choice_rotation) % 360

        print(f"[audio->video] Creating bbox for choice {i} at {choice_azimuth}°")
        # if not os.path.exists(bbox_video_path):
            # add_bbox_to_video(original_video_path, bbox_video_path, choice_azimuth, elevation, event_class)
        choice_video_paths.append(bbox_video_path)

    correct_answer = ['A', 'B', 'C', 'D'][metadata['correct_answer']]

    question = {
        "question": (
            f"Listen to this spatial audio of a {event_class.lower()} sound. "
            f"Which panoramic video view with red bounding box best indicates the direction the sound comes from? "
            f"Choose A, B, C, or D."
        ),
        "conditions": {
            "modality": "Audio",
            "input": make_absolute_path(input_audio_path),
            "description": f"Spatial audio of {event_class.lower()} from {get_azimuth_description(azimuth)}",
        },
        "options": {
            "A": {
                "modality": "Video",
                "input": make_absolute_path(choice_video_paths[0]),
                "description": f"360° view with red bbox, center at {metadata['choice_video_rotations'][0]} degree",
            },
            "B": {
                "modality": "Video",
                "input": make_absolute_path(choice_video_paths[1]),
                "description": f"360° view with red bbox, center at {metadata['choice_video_rotations'][1]} degree",
            },
            "C": {
                "modality": "Video",
                "input": make_absolute_path(choice_video_paths[2]),
                "description": f"360° view with red bbox, center at {metadata['choice_video_rotations'][2]} degree",
            },
            "D": {
                "modality": "Video",
                "input": make_absolute_path(choice_video_paths[3]),
                "description": f"360° view with red bbox, center at {metadata['choice_video_rotations'][3]} degree",
            }
        },
        "correct_answer": correct_answer,
        "extra_info": {
            "event_class": event_class,
            "azimuth": azimuth,
            "question_type": "audio_to_video_bbox",
            "source_instance": selected_instance['question_id'],
            "choice_rotations": metadata['choice_video_rotations'],
            "has_bbox": True
        }
    }
    return question


# ---- 2) Video -> Audio (bbox) ----
def generate_question_video_audio_with_bbox(selected_instance) -> Dict[str, Any]:
    """Generate video to audio question with bbox-enhanced videos"""
    metadata_path = os.path.join(selected_instance['path'], selected_instance['metadata_file'])
    metadata = load_metadata(metadata_path)

    input_video_path = os.path.join(selected_instance['path'], f"{selected_instance['question_id']}_input_video.mp4")
    bbox_input_video_path = os.path.join(selected_instance['path'], f"{selected_instance['question_id']}_input_video.mp4")

    event_class = metadata['event_info']['class']
    azimuth = metadata['event_info']['azimuth']
    elevation = metadata['event_info']['elevation']

    # print(f"[video->audio] Creating bbox for input video at {azimuth}°")
    # if not os.path.exists(bbox_input_video_path):
        # add_bbox_to_video(input_video_path, bbox_input_video_path, azimuth, elevation, event_class)

    choice_audio_paths = [os.path.join(selected_instance['path'], f"{selected_instance['question_id']}_choice_{i}.wav") for i in range(4)]
    correct_answer = ['A', 'B', 'C', 'D'][metadata['correct_answer']]

    question = {
        "question": (
            "Watch this panoramic video view with a red bounding box indicating the expected sound-source direction. "
            "Which spatial audio clip matches this view? Choose A, B, C, or D."
        ),
        "conditions": {
            "modality": "Video",
            "input": make_absolute_path(bbox_input_video_path),
            "description": f"360° video with {event_class.lower()} and red bbox at {azimuth}°",
        },
        "options": {
            "A": {
                "modality": "Audio",
                "input": make_absolute_path(choice_audio_paths[0]),
                "description": f"Spatial audio with rotation {metadata['choice_audio_rotations'][0]} degree",
            },
            "B": {
                "modality": "Audio",
                "input": make_absolute_path(choice_audio_paths[1]),
                "description": f"Spatial audio with rotation {metadata['choice_audio_rotations'][1]} degree",
            },
            "C": {
                "modality": "Audio",
                "input": make_absolute_path(choice_audio_paths[2]),
                "description": f"Spatial audio with rotation {metadata['choice_audio_rotations'][2]} degree",
            },
            "D": {
                "modality": "Audio",
                "input": make_absolute_path(choice_audio_paths[3]),
                "description": f"Spatial audio with rotation {metadata['choice_audio_rotations'][3]} degree",
            }
        },
        "correct_answer": correct_answer,
        "extra_info": {
            "event_class": event_class,
            "azimuth": azimuth,
            "question_type": "video_to_audio_bbox",
            "source_instance": selected_instance['question_id'],
            "choice_rotations": metadata['choice_audio_rotations'],
            "has_bbox": True
        }
    }
    return question


# ---- 3) Audio -> Text ----
def generate_question_audio_text(selected_instance: Dict) -> Dict[str, Any]:
    """Given spatial audio, choose the correct text description (direction)."""
    metadata_path = os.path.join(selected_instance['path'], selected_instance['metadata_file'])
    metadata = load_metadata(metadata_path)

    input_audio_path = os.path.join(selected_instance['path'], f"{selected_instance['question_id']}_input_audio.wav")
    event_class = metadata['event_info']['class']
    azimuth = metadata['event_info']['azimuth']
    correct_idx = metadata['correct_answer']

    # 基于 choice_video_rotations（或 audio_rotations 也可）生成四个文本描述
    # 这里选择使用 choice_video_rotations，对应不同视角/角度的描述语句
    text_options = []
    # import ipdb; ipdb.set_trace()
    for rot in metadata['choice_video_rotations']:
        # 从输入音频的真实方位出发，描述“从当前听者正前方向旋转 rot 后的相对方位”
        desc_az = (azimuth + rot) % 360
        text_options.append(
            f"The {event_class.lower()} sound comes from {get_azimuth_description(desc_az)}."
        )

    correct_answer = ['A', 'B', 'C', 'D'][correct_idx]

    question = {
        "question": (
            f"Listen to the spatial audio. Which text best describes the direction of the {event_class.lower()} sound? (Angles range from -180° to 180°, increasing counter-clockwise: 0° = front, +90° = left, −90° = right, ±180° = back)"
        ),
        "conditions": {
            "modality": "Audio",
            "input": make_absolute_path(input_audio_path),
            "description": f"Spatial audio of {event_class.lower()}",
        },
        "options": {
            "A": {"modality": "Text", "input": text_options[0]},
            "B": {"modality": "Text", "input": text_options[1]},
            "C": {"modality": "Text", "input": text_options[2]},
            "D": {"modality": "Text", "input": text_options[3]},
        },
        "correct_answer": correct_answer,
        "extra_info": {
            "event_class": event_class,
            "azimuth": azimuth,
            "question_type": "audio_to_text",
            "source_instance": selected_instance['question_id'],
            "choice_rotations": metadata['choice_video_rotations'],
        }
    }
    return question


# ---- 4) Video -> Text (bbox) ----
def generate_question_video_text(selected_instance: Dict) -> Dict[str, Any]:
    """Given 360° video with bbox, choose the correct text description (direction)."""
    metadata_path = os.path.join(selected_instance['path'], selected_instance['metadata_file'])
    metadata = load_metadata(metadata_path)

    input_video_path = os.path.join(selected_instance['path'], f"{selected_instance['question_id']}_input_video.mp4")
    bbox_input_video_path = os.path.join(selected_instance['path'], f"{selected_instance['question_id']}_input_video.mp4")

    event_class = metadata['event_info']['class']
    azimuth = metadata['event_info']['azimuth']
    elevation = metadata['event_info']['elevation']
    correct_idx = metadata['correct_answer']

    # 输入视频画 bbox
    print(f"[video->text] Creating bbox for input video at {azimuth}°")
    # if not os.path.exists(bbox_input_video_path):
        # add_bbox_to_video(input_video_path, bbox_input_video_path, azimuth, elevation, event_class)

    # 四个文本选项：围绕不同 rotation 的方向描述
    text_options = []
    for rot in metadata['choice_audio_rotations']:
        # 对于给定视频（红框在 azimuth），若观察者旋转 rot，则声音相对描述为：
        desc_az = (azimuth + rot) % 360
        text_options.append(
            f"The {event_class.lower()} sound is {get_azimuth_description(desc_az)}."
        )

    correct_answer = ['A', 'B', 'C', 'D'][correct_idx]

    question = {
        "question": (
            "Watch the panoramic video with a red bounding box indicating the expected sound source. "
            "Which text best matches the indicated direction? (The middle of the image is front. Angles range from -180° to 180°, increasing counter-clockwise: 0° = front, +90° = left, −90° = right, ±180° = back)"
        ),
        "conditions": {
            "modality": "Video",
            "input": make_absolute_path(bbox_input_video_path),
            "description": f"Video with red bbox at {azimuth}°",
        },
        "options": {
            "A": {"modality": "Text", "input": text_options[0]},
            "B": {"modality": "Text", "input": text_options[1]},
            "C": {"modality": "Text", "input": text_options[2]},
            "D": {"modality": "Text", "input": text_options[3]},
        },
        "correct_answer": correct_answer,
        "extra_info": {
            "event_class": event_class,
            "azimuth": azimuth,
            "question_type": "video_to_text_bbox",
            "source_instance": selected_instance['question_id'],
            "choice_rotations": metadata['choice_audio_rotations'],
            "has_bbox": True
        }
    }
    return question


# ---- 5) Text -> Audio ----
def generate_question_text_audio(selected_instance: Dict) -> Dict[str, Any]:
    """Given a text description, choose the matching spatial audio."""
    metadata_path = os.path.join(selected_instance['path'], selected_instance['metadata_file'])
    metadata = load_metadata(metadata_path)

    event_class = metadata['event_info']['class']
    azimuth = metadata['event_info']['azimuth']
    correct_idx = metadata['correct_answer']

    # 文本条件直接描述真实方向
    text_input = (
        f"A {event_class.lower()} sound is {get_azimuth_description(azimuth)}. "
    )

    # 四个音频选项
    choice_audio_paths = [os.path.join(selected_instance['path'], f"{selected_instance['question_id']}_choice_{i}.wav") for i in range(4)]
    correct_answer = ['A', 'B', 'C', 'D'][correct_idx]

    question = {
        "question": "Select the spatial audio clip that best matches the text-described sound direction. (Angles range from -180° to 180°, increasing counter-clockwise: 0° = front, +90° = left, −90° = right, ±180° = back)",
        "conditions": {
            "modality": "Text",
            "input": text_input,
            "description": f"{event_class.lower()} sound direction description",
        },
        "options": {
            "A": {
                "modality": "Audio",
                "input": make_absolute_path(choice_audio_paths[0]),
                "description": f"Spatial audio with rotation {metadata['choice_audio_rotations'][0]} degree",
            },
            "B": {
                "modality": "Audio",
                "input": make_absolute_path(choice_audio_paths[1]),
                "description": f"Spatial audio with rotation {metadata['choice_audio_rotations'][1]} degree",
            },
            "C": {
                "modality": "Audio",
                "input": make_absolute_path(choice_audio_paths[2]),
                "description": f"Spatial audio with rotation {metadata['choice_audio_rotations'][2]} degree",
            },
            "D": {
                "modality": "Audio",
                "input": make_absolute_path(choice_audio_paths[3]),
                "description": f"Spatial audio with rotation {metadata['choice_audio_rotations'][3]} degree",
            }
        },
        "correct_answer": correct_answer,
        "extra_info": {
            "event_class": event_class,
            "azimuth": azimuth,
            "question_type": "text_to_audio",
            "source_instance": selected_instance['question_id'],
            "choice_rotationsf": metadata['choice_audio_rotations'],
        }
    }
    return question


# ---- 6) Text -> Video (bbox) ----
def generate_question_text_video(selected_instance: Dict) -> Dict[str, Any]:
    """Given a text description, choose the matching 360° video view with bbox."""
    metadata_path = os.path.join(selected_instance['path'], selected_instance['metadata_file'])
    metadata = load_metadata(metadata_path)

    event_class = metadata['event_info']['class']
    azimuth = metadata['event_info']['azimuth']
    elevation = metadata['event_info']['elevation']
    correct_idx = metadata['correct_answer']

    # 文本条件直接描述真实方向
    text_input = (
        f"A {event_class.lower()} sound is {get_azimuth_description(azimuth)}. "
    )

    # 四个视频选项（加 bbox）
    choice_video_paths = []
    for i in range(4):
        original_video_path = os.path.join(selected_instance['path'], f"{selected_instance['question_id']}_choice_{i}.mp4")
        bbox_video_path = os.path.join(selected_instance['path'], f"{selected_instance['question_id']}_choice_{i}.mp4")

        choice_rotation = metadata['choice_video_rotations'][i]
        choice_azimuth = (azimuth + choice_rotation) % 360

        print(f"[text->video] Creating bbox for choice {i} at {choice_azimuth}°")
        # if not os.path.exists(bbox_video_path): 
            # add_bbox_to_video(original_video_path, bbox_video_path,  choice_azimuth, elevation, event_class)
        choice_video_paths.append(bbox_video_path)

    correct_answer = ['A', 'B', 'C', 'D'][correct_idx]

    question = {
        "question": "Select the panoramic video view with red bbox that matches the text-described direction. (The middle of the image is front. Angles range from -180° to 180°, increasing counter-clockwise: 0° = front, +90° = left, −90° = right, ±180° = back)",
        "conditions": {
            "modality": "Text",
            "input": text_input,
            "description": f"{event_class.lower()} direction description",
        },
        "options": {
            "A": {
                "modality": "Video",
                "input": make_absolute_path(choice_video_paths[0]),
                "description": f"360° view with red bbox, center at {metadata['choice_video_rotations'][0]} degree",
            },
            "B": {
                "modality": "Video",
                "input": make_absolute_path(choice_video_paths[1]),
                "description": f"360° view with red bbox, center at {metadata['choice_video_rotations'][1]} degree",
            },
            "C": {
                "modality": "Video",
                "input": make_absolute_path(choice_video_paths[2]),
                "description": f"360° view with red bbox, center at {metadata['choice_video_rotations'][2]} degree",
            },
            "D": {
                "modality": "Video",
                "input": make_absolute_path(choice_video_paths[3]),
                "description": f"360° view with red bbox, center at {metadata['choice_video_rotations'][3]} degree",
            }
        },
        "correct_answer": correct_answer,
        "extra_info": {
            "event_class": event_class,
            "azimuth": azimuth,
            "question_type": "text_to_video_bbox",
            "source_instance": selected_instance['question_id'],
            "choice_rotations": metadata['choice_video_rotations'],
            "has_bbox": True
        }
    }
    return question


# ==================== Main ====================

def main():
    random.seed(42)  # For reproducibility

    DATASET_NAME = 'starss23'

    # Base directories
    video_choice_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/STARSS23_processed_augmented/questions_video_choice"  # 输入是 Audio，选项是 Video
    audio_choice_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/STARSS23_processed_augmented/questions_audio_choice"  # 输入是 Video，选项是 Audio

    max_questions_per_type = 500

    export_dir = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/02_spatial/panaroma'
    os.makedirs(export_dir, exist_ok=True)

    # Initialize question lists
    audio_video_questions = []  # Audio -> Video
    video_audio_questions = []  # Video -> Audio
    audio_text_questions = []   # Audio -> Text
    video_text_questions = []   # Video -> Text
    text_audio_questions = []   # Text -> Audio
    text_video_questions = []   # Text -> Video

    all_video_instances = []  # 来自 video_choice_dir（Audio->Video 等）
    all_audio_instances = []  # 来自 audio_choice_dir（Video->Audio 等）
    
    # high_quality_events = []
    # with open('/home/xwang378/scratch/2025/AudioBench/benchmark/Data/STARSS23_processed_augmented/high_quality.txt', 'r') as f:
    #     for line in f:
    #         high_quality_events.append(line.strip())

    # Collect instances
    print("Collecting video choice instances...")
    if os.path.exists(video_choice_dir):
        for split in ['dev-test-sony', 'dev-test-tau']:
            split_dir = os.path.join(video_choice_dir, split)
            if os.path.isdir(split_dir):
                for clip in os.listdir(split_dir):
                    clip_dir = os.path.join(split_dir, clip)
                    if os.path.isdir(clip_dir):
                        for file in os.listdir(clip_dir):
                            if file.endswith('_video_choice_metadata.json') or file.endswith('_video_choice_metadata_add_on.json'):
                                question_id = file.replace('_metadata.json', '').replace('_metadata_add_on.json', '')

                                instance_info = {
                                    'path': clip_dir,
                                    'question_id': question_id,
                                    'metadata_file': file,
                                    'type': 'video_choice'
                                }
                                all_video_instances.append(instance_info)

    print("Collecting audio choice instances...")
    if os.path.exists(audio_choice_dir):
        for split in ['dev-test-sony', 'dev-test-tau']:
            split_dir = os.path.join(audio_choice_dir, split)
            if os.path.isdir(split_dir):
                for clip in os.listdir(split_dir):
                    clip_dir = os.path.join(split_dir, clip)
                    if os.path.isdir(clip_dir):
                        for file in os.listdir(clip_dir):
                            if file.endswith('_audio_choice_metadata_add_on.json') or file.endswith('_audio_choice_metadata.json'):
                                question_id = file.replace('_metadata.json', '').replace('_metadata_add_on.json', '')
                                instance_info = {
                                    'path': clip_dir,
                                    'question_id': question_id,
                                    'metadata_file': file,
                                    'type': 'audio_choice'
                                }
                                all_audio_instances.append(instance_info)
    print(f"Found {len(all_video_instances)} video choice instances")
    print(f"Found {len(all_audio_instances)} audio choice instances")

    # Limit the number of instances
    if len(all_video_instances) > max_questions_per_type:
        all_video_instances = random.sample(all_video_instances, max_questions_per_type)
    if len(all_audio_instances) > max_questions_per_type:
        all_audio_instances = random.sample(all_audio_instances, max_questions_per_type)

    # ---- Process video_choice instances (Audio -> Video / Audio -> Text / Text -> Video) ----
    print(f"\n=== Processing Video Choice Instances (Audio->Video, Audio->Text, Text->Video) ===")
    for i, inst in enumerate(all_video_instances):
        try:
            q_audio_video = generate_question_audio_video_with_bbox(inst)
            q_audio_text = generate_question_audio_text(inst)
            q_text_video = generate_question_text_video(inst)

            audio_video_questions.append(q_audio_video)
            audio_text_questions.append(q_audio_text)
            text_video_questions.append(q_text_video)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(all_video_instances)} video-choice instances")
        except Exception as e:
            print(f"[ERROR] video-choice {inst['question_id']}: {e}")
            continue

    # ---- Process audio_choice instances (Video -> Audio / Video -> Text / Text -> Audio) ----
    print(f"\n=== Processing Audio Choice Instances (Video->Audio, Video->Text, Text->Audio) ===")
    for i, inst in enumerate(all_audio_instances):
        try:
            q_video_audio = generate_question_video_audio_with_bbox(inst)
            q_video_text = generate_question_video_text(inst)
            q_text_audio = generate_question_text_audio(inst)

            video_audio_questions.append(q_video_audio)
            video_text_questions.append(q_video_text)
            text_audio_questions.append(q_text_audio)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(all_audio_instances)} audio-choice instances")
        except Exception as e:
            print(f"[ERROR] audio-choice {inst['question_id']}: {e}")
            continue

    # ---- Export all six tasks ----
    print(f"\n=== Exporting All Six Tasks ===")
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_audio_video.json", "w") as f:
        json.dump(audio_video_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_video_audio.json", "w") as f:
        json.dump(video_audio_questions, f, indent=4)

    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_audio_text.json", "w") as f:
        json.dump(audio_text_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_video_text.json", "w") as f:
        json.dump(video_text_questions, f, indent=4)

    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_text_audio.json", "w") as f:
        json.dump(text_audio_questions, f, indent=4)
    with open(f"{export_dir}/{DATASET_NAME}_audio_bench_questions_text_video.json", "w") as f:
        json.dump(text_video_questions, f, indent=4)

    # ---- Summary ----
    print(f"\n=== Final Summary ===")
    print(f"[Audio->Video] {len(audio_video_questions)}")
    print(f"[Video->Audio] {len(video_audio_questions)}")
    print(f"[Audio->Text ] {len(audio_text_questions)}")
    print(f"[Video->Text ] {len(video_text_questions)}")
    print(f"[Text->Audio ] {len(text_audio_questions)}")
    print(f"[Text->Video ] {len(text_video_questions)}")
    total = (len(audio_video_questions) + len(video_audio_questions) +
             len(audio_text_questions) + len(video_text_questions) +
             len(text_audio_questions) + len(text_video_questions))
    print(f"Total: {total} questions")
    print(f"Exported to: {export_dir}")


if __name__ == "__main__":
    main()
