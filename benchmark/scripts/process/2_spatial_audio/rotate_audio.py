import numpy as np


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
    if audio_mic.ndim != 2:
        raise ValueError("audio_mic应为 [M, T]，通道在前")
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
