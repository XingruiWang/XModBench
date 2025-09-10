#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
quad_binaural.py — 4-ch mic array -> binaural WAV via SOFA HRTF.

Features:
- Mic-aware: supply (azimuth°, elevation°) for each channel.
- Uses SOFA (HRIR time-domain or HRTF freq-domain).
- Optional near-field delay from mic radius (seconds ~ r / c).
- Robust deg/rad handling, azimuth wrap, safe peak trim.
- CLI usage or inline defaults for your M1..M4 layout.

Example:
  python quad_binaural.py \
    --in your_4ch.wav \
    --sofa subject_008.sofa \
    --out binaural.wav \
    --dirs "[(45,35),(-45,-35),(135,-35),(-135,35)]" \
    --peak_dbfs -1

Or with JSON mapping:
  python quad_binaural.py \
    --in your_4ch.wav \
    --sofa subject_008.sofa \
    --out binaural.wav \
    --dirs_json mic_dirs.json

Where mic_dirs.json contains:
  {"dirs_deg": [[45,35],[-45,-35],[135,-35],[-135,35]], "radius_m": 0.042}
"""

import ast
import json
import argparse
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
import pysofaconventions

# ------------------------ Utilities ------------------------


def enhance_front_back(L, R, az_deg, sr):
    """增强前后位置的频谱特征差异"""
    from scipy.signal import butter, filtfilt
    
    # 后方声源添加轻微高频衰减（模拟头部遮挡）
    if abs(az_deg) > 90:  # 后方
        b, a = butter(2, 8000/(sr/2), 'low')
        L = filtfilt(b, a, L) * 0.6
        R = filtfilt(b, a, R) * 0.6
    
    # 前方声源增强中高频清晰度
    else:  # 前方
        b, a = butter(2, [2000/(sr/2), 6000/(sr/2)], 'band')
        enhancement = filtfilt(b, a, L) * 0.2
        L = L + enhancement
        enhancement = filtfilt(b, a, R) * 0.2
        R = R + enhancement
    
    return L, R

def wrap180(x_deg: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x_deg) + 180.0) % 360.0 - 180.0

def as_degrees(az: np.ndarray, el: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    az = az.astype(float); el = el.astype(float)
    # Heuristic: if mostly within +/-3.2, assume radians -> convert
    if np.percentile(np.abs(az), 95) < 3.2 and np.percentile(np.abs(el), 95) < 3.2:
        az = np.degrees(az); el = np.degrees(el)
    return az, el

def get_IR_MRN(sofa) -> np.ndarray:
    """
    Return HRIR array of shape [M, 2, N].
    If missing Data.IR, reconstruct from (Data.Real, Data.Imag).
    """
    IR = sofa.getDataIR()
    if IR is not None and np.size(IR) > 0:
        return IR
    real = sofa.getDataReal()
    imag = sofa.getDataImag()
    if real is None or imag is None:
        raise ValueError("SOFA lacks Data.IR and (Data.Real, Data.Imag).")
    H = real + 1j * imag  # [M, 2, F]
    IR = np.fft.irfft(H, axis=-1)  # -> [M, 2, N]
    return IR

def nearest_idx_for(sofa, az_deg_tgt: float, el_deg_tgt: float) -> int:
    """
    Find nearest SOFA measurement index by (az, el) in degrees.
    """
    pos = sofa.getVariableValue('SourcePosition')  # [M, 3] (az, el, r)
    az = pos[:, 0].astype(float)
    el = pos[:, 1].astype(float)
    az, el = as_degrees(az, el)
    # Use wrap for azimuth difference
    d_az = wrap180(az - az_deg_tgt)
    d_el = (el - el_deg_tgt)
    idx = int(np.argmin(d_az**2 + d_el**2))
    return idx

def get_hrir(sofa, az_deg: float, el_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    IR = get_IR_MRN(sofa)  # [M, 2, N], receivers: 0=Left,1=Right
    idx = nearest_idx_for(sofa, az_deg, el_deg)
    hL = IR[idx, 0, :]
    hR = IR[idx, 1, :]
    if not np.any(np.abs(hL)) and not np.any(np.abs(hR)):
        raise ValueError(f"HRIR at idx={idx} (az={az_deg}, el={el_deg}) is all zeros.")
    return hL, hR

def convolve_to_stereo(x_mono: np.ndarray, hL: np.ndarray, hR: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L = fftconvolve(x_mono, hL, mode="full")
    R = fftconvolve(x_mono, hR, mode="full")
    return L, R

def apply_delay_samples(x: np.ndarray, samples: int) -> np.ndarray:
    if samples == 0:
        return x
    if samples > 0:
        return np.pad(x, (samples, 0))
    else:
        return x[-samples:]  # shift earlier by trimming front if negative

def nearfield_delay_seconds(r_m: float, c: float = 343.0) -> float:
    """
    Very small TOF delay ~ r/c. For r=4.2cm -> ~0.000122 s (~5.9 samples @ 48k).
    This is optional and subtle; it slightly decorrelates channels.
    """
    return r_m / c

# ------------------------ Core Renderer ------------------------

def render_4ch_binaural(
    wav_4ch_path: str,
    sofa_path: str,
    dirs_deg: list[tuple[float, float]],  # [(az_deg, el_deg)] per channel, in file order
    out_path: str = "binaural.wav",
    peak_dbfs: float = -1.0,
    use_nearfield_delay: bool = False,
    radius_m: float = 0.042
) -> tuple[str, int, np.ndarray]:
    """
    Convert 4-ch mic WAV -> binaural stereo WAV using SOFA HRTF.

    dirs_deg: 4 pairs matching channel order, e.g.
      [(45,35), (-45,-35), (135,-35), (-135,35)]
      for your M1..M4 array.
    """
    # 1) load multichannel WAV
    x, sr = sf.read(wav_4ch_path)
    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError(f"Expected 4-channel WAV, got shape {x.shape}")
    chs = [x[:, i] for i in range(4)]

    # 2) open SOFA
    sofa = pysofaconventions.SOFAFile(sofa_path, "r")

    # 3) optional small near-field delay per channel (same delay for all here; could vary by az if needed)
    delay_samps = 0
    if use_nearfield_delay:
        dt = nearfield_delay_seconds(radius_m)            # ~0.000122 s
        delay_samps = int(round(dt * sr))                 # small integer samples
        # NOTE: you can vary delay sign by azimuth if you model head center;
        # here we just add a tiny positive delay to all channels for decorrelation.

    # 4) render per channel
    L_sum = None
    R_sum = None
    for ch, (az, el) in zip(chs, dirs_deg):
        # apply tiny near-field delay (optional)
        chd = apply_delay_samples(ch, delay_samps) if delay_samps else ch
        hL, hR = get_hrir(sofa, float(az), float(el))
        Lc, Rc = convolve_to_stereo(chd, hL, hR)
        Lc, Rc = enhance_front_back(Lc, Rc, az, sr)
        if L_sum is None:
            L_sum, R_sum = Lc, Rc
        else:
            # pad to same length then sum
            n = max(len(L_sum), len(Lc))
            if len(L_sum) < n: L_sum = np.pad(L_sum, (0, n - len(L_sum)))
            if len(R_sum) < n: R_sum = np.pad(R_sum, (0, n - len(R_sum)))
            if len(Lc)    < n: Lc    = np.pad(Lc,    (0, n - len(Lc)))
            if len(Rc)    < n: Rc    = np.pad(Rc,    (0, n - len(Rc)))
            L_sum += Lc
            R_sum += Rc

    # 5) peak normalize to target
    peak = float(max(np.max(np.abs(L_sum)), np.max(np.abs(R_sum)), 1e-12))
    target = 10.0 ** (peak_dbfs / 20.0)  # -1 dBFS -> ~0.891
    gain = min(1.0, target / peak)
    L_out = (L_sum * gain).astype(np.float32)
    R_out = (R_sum * gain).astype(np.float32)
    y = np.stack([L_out, R_out], axis=1)

    # 6) write
    sf.write(out_path, y, sr)
    return out_path, sr, y

# ------------------------ CLI ------------------------

def parse_dirs_arg(dirs_str: str) -> list[tuple[float, float]]:
    """
    Parse --dirs string like "[(45,35),(-45,-35),(135,-35),(-135,35)]"
    into a list of tuples.
    """
    val = ast.literal_eval(dirs_str)
    if len(val) != 4:
        raise ValueError("dirs must contain exactly 4 pairs.")
    return [(float(a), float(b)) for a, b in val]

def main():
    ap = argparse.ArgumentParser(description="4-ch mic -> binaural via SOFA HRTF")
    ap.add_argument("--in", dest="inp", required=False, default="your_4ch.wav", help="Input 4-ch WAV")
    ap.add_argument("--sofa", required=False, default="subject_008.sofa", help="SOFA (HRIR/HRTF)")
    ap.add_argument("--out", required=False, default="binaural.wav", help="Output stereo WAV")
    ap.add_argument("--dirs", required=False,
                    default="[(45,35),(-45,-35),(135,-35),(-135,35)]",
                    help="4 pairs of (az_deg, el_deg) matching channel order")
    ap.add_argument("--dirs_json", required=False, help='JSON path with {"dirs_deg": [[az,el],...], "radius_m": 0.042}')
    ap.add_argument("--peak_dbfs", type=float, default=-1.0, help="Output peak target dBFS (e.g., -1)")
    ap.add_argument("--nearfield", action="store_true", help="Enable tiny near-field delay from radius")
    ap.add_argument("--radius_m", type=float, default=0.042, help="Mic radius in meters if --nearfield")
    args = ap.parse_args()

    if args.dirs_json:
        with open(args.dirs_json, "r") as f:
            cfg = json.load(f)
        dirs_deg = [(float(a), float(b)) for (a, b) in cfg["dirs_deg"]]
        radius_m = float(cfg.get("radius_m", args.radius_m))
    else:
        dirs_deg = parse_dirs_arg(args.dirs)
        radius_m = args.radius_m

    out_path, sr, _ = render_4ch_binaural(
        wav_4ch_path=args.inp,
        sofa_path=args.sofa,
        dirs_deg=dirs_deg,
        out_path=args.out,
        peak_dbfs=args.peak_dbfs,
        use_nearfield_delay=args.nearfield,
        radius_m=radius_m
    )
    print(f"[done] wrote {out_path} @ {sr} Hz")

if __name__ == "__main__":
    main()

