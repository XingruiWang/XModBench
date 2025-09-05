import numpy as np
from typing import List, Tuple
import subprocess
import os

def _foa_basis_from_dirs(phi, theta):
    """
    你给的FOA基函数（实数、频率无关近似）：
    H1=1, H2=sin(phi)*cos(theta), H3=sin(theta), H4=cos(phi)*cos(theta)
    返回 Y: [M, 4]，列顺序 = [W, Y, Z, X]
    """
    H1 = np.ones_like(phi)
    H2 = np.sin(phi) * np.cos(theta)  # Y (L-R)
    H3 = np.sin(theta)                # Z (U-D)
    H4 = np.cos(phi) * np.cos(theta)  # X (F-B)
    Y = np.stack([H1, H2, H3, H4], axis=1)
    return Y

def rotate_audio_azimuth_mic(
    audio_mic: np.ndarray,
    mic_dirs: np.ndarray,
    rotation_degrees: float,
    gW: float = 1.0, gXY: float = 1.0, gZ: float = 1.0,
    equal_loudness: bool = True, headroom_db: float = 1.0
) -> np.ndarray:
    """
    用 FOA 近似把“麦克风阵列信号”整体在水平面旋转（yaw/azimuth），
    输入/输出依然是 mic 格式（形状 [M, T]）。

    约定：rotation_degrees > 0 表示 **逆时针 (CCW)**。
    使用你给的 FOA 基函数（频率无关近似；~9 kHz 内更合理）。

    参数
    ----
    audio_mic : np.ndarray
        [M, T]，M个麦，T个采样；通道在前
    mic_dirs : np.ndarray
        [M, 2]，每个麦的方向角 (phi, theta)，单位：弧度
        phi=方位角(0..2π)，theta=仰角(-π/2..+π/2)
    rotation_degrees : float
        逆时针旋转角度（度）
    gW, gXY, gZ : float
        可选的通道增益（增强方向感时可适当降低W、提高XY）
    equal_loudness : bool
        是否做全局等响度（能量/RMS）归一化
    headroom_db : float
        归一化后预留余量，防止削波

    返回
    ----
    audio_mic_rot : np.ndarray
        [M, T]，旋转后的阵列信号（仍为mic格式）
    """
    # if T, M shape

    if audio_mic.ndim != 2:
        raise ValueError("audio_mic应为 [M, T]，通道在前")
    if audio_mic.shape[0] > 100:
        audio_mic = np.transpose(audio_mic, (1, 0))
    M, T = audio_mic.shape
    if mic_dirs.shape != (M, 2):
        raise ValueError("mic_dirs 形状需为 [M, 2]，列为 (phi, theta)，单位弧度")

    # 0) 基线响度
    if equal_loudness:
        rms_in = np.sqrt(np.mean(audio_mic**2) + 1e-12)

    # 1) FOA 编码（最小二乘，频率无关近似）
    phi = mic_dirs[:, 0]
    theta = mic_dirs[:, 1]
    Y = _foa_basis_from_dirs(phi, theta)          # [M, 4]，列序 [W, Y, Z, X]
    # 预计算伪逆：A = pinv(Y) @ X  =>  X≈Y A
    # pinv(Y) = (Y^T Y)^(-1) Y^T
    YtY = Y.T @ Y                                  # [4,4]
    Y_pinv = np.linalg.pinv(YtY) @ Y.T             # [4,M]
    A = Y_pinv @ audio_mic                         # [4, T]  -> FOA系数: [W, Y, Z, X]

    # 2) 在 FOA 域做 CCW 旋转（只旋转 X/Y；W,Z 不变）
    #    我们现在通道顺序是 [W, Y, Z, X]（与你给的H1..H4一致）
    W, Yc, Z, X = 0, 1, 2, 3
    theta_deg = rotation_degrees
    th = np.radians(theta_deg)
    c, s = np.cos(th), np.sin(th)

    X_in = A[X, :].copy()
    Y_in = A[Yc, :].copy()
    A[X, :]  =  c * X_in - s * Y_in
    A[Yc, :] =  s * X_in + c * Y_in
    # W,Z unchanged

    # 3) 可选：增强方向感的通道增益（谨慎）
    if (gW != 1.0) or (gXY != 1.0) or (gZ != 1.0):
        A[W, :]  *= gW
        A[X, :]  *= gXY
        A[Yc, :] *= gXY
        A[Z, :]  *= gZ

    # 4) 解码回阵列：X_rot ≈ Y @ A_rot
    audio_mic_rot = Y @ A                          # [M, T]

    # 5) 等响度归一化 + 预留余量
    if equal_loudness:
        rms_out = np.sqrt(np.mean(audio_mic_rot**2) + 1e-12)
        if rms_out > 0:
            target = rms_in * (10 ** (-headroom_db / 20.0))
            audio_mic_rot *= (target / rms_out)
        # 防削波
        peak = np.max(np.abs(audio_mic_rot))
        if peak > 0.999:
            audio_mic_rot *= (0.999 / peak)

    return audio_mic_rot

# def rotate_video_azimuth(video_data, rotation_degrees):
import cv2

def create_centered_360_view(frame: np.ndarray, center_azimuth: float, 
                            output_width: int = 800, output_height: int = 400) -> np.ndarray:
    """
    Create a 360° view by horizontally shifting the equirectangular image 
    so that center_azimuth appears in the middle
    """
    height, width = frame.shape[:2]
    # Convert azimuth to pixel offset
    # Azimuth 0° should be at width/2, azimuth 180° at 0, azimuth -180° at width
    azimuth_normalized = (center_azimuth % 360) / 360.0  # 0 to 1
    pixel_offset = int(azimuth_normalized * width)
    
    # Create shifted image by rolling horizontally
    shifted_frame = np.roll(frame, -pixel_offset, axis=1)
    
    # Resize to desired output size
    resized_frame = cv2.resize(shifted_frame, (output_width, output_height))
    
    return resized_frame

def rotate_video_azimuth(video_path: str, center_azimuth: float, output_size: Tuple[int, int] = (1920, 960)) -> List[np.ndarray]:
    """Extract video segment with specific center azimuth"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame range
    start_frame = 0
    end_frame = 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    
    frames = []
    for frame_idx in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Apply 360° transformation with center azimuth
            transformed_frame = create_centered_360_view(frame, center_azimuth, 
                                                            output_size[0], output_size[1])
            frames.append(transformed_frame)
        else:
            break
    
    cap.release()
    return frames


    
def save_video_clip(frames: List[np.ndarray], output_path: str, fps: float = 30.0):
    """Save frames as video clip"""
    if not frames:
        raise ValueError("No frames to save")
    
    output_path = str(output_path)
    temp_path = str(output_path).replace('.mp4', '_tmp.mp4')
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 改为 H264 编码
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
        
    out.release()
    
    # Re-encode to H.264 using ffmpeg
    cmd = [
        "ffmpeg", "-y", "-i", temp_path,
        "-c:v", "libx264", "-crf", "23", "-preset", "medium",
        "-an",  # 音频用AAC（如果没有音频轨道也没问题）
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=False)

    # 删除临时文件
    os.remove(temp_path)
    