#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 2024-2025 流行歌手数据集（图片 + 手动指定YouTube音频，带详细日志 & 并行）
- 图片：Wikimedia Commons（开放许可） + Wikipedia 首图（如已存在则跳过）
- 音频：手动指定 YouTube 视频ID，自动下载转换为 wav
- 不做音频切分
- 并行：不同歌手并行处理；yt-dlp 并发可单独限速
仅限学术研究。请遵守站点条款与版权，不要公开分发受版权保护的音频。
"""

import os, re, json, time, argparse, pathlib, subprocess, shutil, threading, traceback
from typing import List, Dict, Optional
import concurrent.futures as futures

import requests
import pandas as pd

# ------------------------ 常量配置 ------------------------
UA = {"User-Agent": "PopSingerDataset/YouTube-1.2 (research; contact: example@example.com)"}
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

ALLOW_LICENSE = (
    "cc-zero","cc0","cc by","cc-by","cc-by-sa","cc-by-nd","cc-by-nc","cc-by-nc-sa","cc-by-nc-nd",
    "public domain","pd"
)
IMG_KEYWORDS = ("portrait","headshot","singer","vocalist","live","concert","press","promotional")

POPULAR_SINGERS_2024_2025 = [
    {"name": "Jay Chou", "chinese_name": "周杰伦", "region": "Taiwan", "genre": "Mandopop"},
    {"name": "JJ Lin", "chinese_name": "林俊杰", "region": "Singapore", "genre": "Mandopop"},
    {"name": "David Tao", "chinese_name": "陶喆", "region": "Taiwan", "genre": "R&B"},
    {"name": "Faye Wong", "chinese_name": "王菲", "region": "Hong Kong", "genre": "Cantopop"},
    {"name": "Eason Chan", "chinese_name": "陈奕迅", "region": "Hong Kong", "genre": "Cantopop"},
    {"name": "Mayday", "chinese_name": "五月天", "region": "Taiwan", "genre": "Rock"},
    {"name": "G.E.M.", "chinese_name": "邓紫棋", "region": "Mainland", "genre": "Pop"},
    {"name": "Joker Xue", "chinese_name": "薛之谦", "region": "Mainland", "genre": "Pop"},
    {"name": "Zhou Shen", "chinese_name": "周深", "region": "Mainland", "genre": "Pop"},
    {"name": "Taylor Swift", "chinese_name": "", "region": "USA", "genre": "Pop"},
    {"name": "Sabrina Carpenter", "chinese_name": "", "region": "USA", "genre": "Pop"},
    {"name": "Billie Eilish", "chinese_name": "", "region": "USA", "genre": "Alt-Pop"},
    {"name": "Ariana Grande", "chinese_name": "", "region": "USA", "genre": "Pop"},
    {"name": "Olivia Rodrigo", "chinese_name": "", "region": "USA", "genre": "Pop"},
    {"name": "The Weeknd", "chinese_name": "", "region": "Canada", "genre": "R&B"},
    {"name": "Bruno Mars", "chinese_name": "", "region": "USA", "genre": "Pop"},
    {"name": "Harry Styles", "chinese_name": "", "region": "UK", "genre": "Pop-Rock"},
    {"name": "Bad Bunny", "chinese_name": "", "region": "Puerto Rico", "genre": "Reggaeton"},
    {"name": "Lady Gaga", "chinese_name": "", "region": "USA", "genre": "Pop"},
    {"name": "Kendrick Lamar", "chinese_name": "", "region": "USA", "genre": "Hip-Hop"},
    {"name": "SZA", "chinese_name": "", "region": "USA", "genre": "R&B"},
    {"name": "Chappell Roan", "chinese_name": "", "region": "USA", "genre": "Alt-Pop"},
    {"name": "Teddy Swims", "chinese_name": "", "region": "USA", "genre": "Soul"},
    {"name": "Morgan Wallen", "chinese_name": "", "region": "USA", "genre": "Country-Pop"},
]

# ------------------------ 手动指定的 YouTube 视频ID ------------------------
# 请在这里手动添加每位歌手的 YouTube 视频ID
MANUAL_YOUTUBE_IDS = {
    "jay chou": [
        # 请在这里添加周杰伦的YouTube视频ID，例如：
        # "dQw4w9WgXcQ",  # 示例ID：歌曲名称
        # "abc123def45",  # 添加更多ID
        "bu7nU9Mhpyo",
        "DYptgVvkVLQ",
        "Z8Mqw0b9ADs",
        "YJfHuATJYsQ",
        "Bbp9ZaJD_eA",
        "sHD_z90ZKV0",
        "1hI-7vj2FhE",
        "qzwsQTY-99o",
        "L229QDxDakU",
        "kfXdP7nZIiE"
        
    ],
    "jj lin": [
        # 请在这里添加林俊杰的YouTube视频ID
        "gd38-X3HpbM",
        "LWV-f6dMN3Q",
        "iE0l8Tx62DE",
        "G97_rOdHcnY",
        "TRt4Y6c0ql0",
        "UsLDdteZSVc",
        "27R6ZavdzzQ",
        "JwjBbWQs71k",
        "7xgucJEqNDo",
        "YFr6p7vB9hc",
        "DrBQeUOdQ_Y"
    ],
    "david tao": [
        # 请在这里添加陶喆的YouTube视频ID
        "Pq5Fxa7tZjM",
        "3L3Me4JXVqE",
        "lhtaCBSL9AE",
        "GurozK-HRTw",
        "2xAmQ4y44eo",
        "dD4JHeZNvF4",
        "yEtU5tzZMkY",
        "1xZede32V6c"
    ],
    "faye wong": [
        # 请在这里添加王菲的YouTube视频ID
        "hN2jOHeI5tc",
        "5wmfXve11rM",
        "04cHqPMD4So",
        "US54FpncMz4",
        "6fV2dRqJHvw",
        "9IORN1EiPP4",
        "lFUUVItSCpk"
    ],
    "eason chan": [
        # 请在这里添加陈奕迅的YouTube视频ID
        "IzKQBo1gLYg",
        "9wLAWuX1FTM",
        "LKMGlj3Y_Lg",
        "bWxHgHLJlYw",
        "p8kyyGCEMCY",
        "bJHCgo85irM",
        "mg1Fy5lRbWo"
    ],
    "mayday": [
        # 请在这里添加五月天的YouTube视频ID
        "d9ktAt-Gg2k",
        "R2s-H_crYkc",
        "38lcQsEMGrk",
        "pd3eV-SG23E",
        "rS8HqJy1UPs"
    ],
    "g.e.m.": [
        # 请在这里添加邓紫棋的YouTube视频ID
        "T4SimnaiktU",
        "Lhel0tzHE08",
        "ma7r2HGqwXs",
        "FWtbGkpdoP4",
        "4OqXWzekVw4",
        "GHXr4bBxHCo",
        "2BEFukvLZfI"
    ],
    "joker xue": [
        # 请在这里添加薛之谦的YouTube视频ID
        "XaN3kUz4KSw",
        "jwkd9wh59us",
        "Ln_94OO-yDc",
        "rK1G8_E8Lb0",
        "SfE6cG4KfHA",
        "rD2IXdpwmkc"
        
    ],
    "zhou shen": [
        # 请在这里添加周深的YouTube视频ID
        "zPYPUYsxU1s",
        "cS4dJvAXiIE",
        "3yW6b4lXnF0",
        "Rk_KPf934aA",
        "W_OK-piGNoQ",
        "HHxAFkB3bA4"
        
    ],
    "taylor swift": [
        # 请在这里添加Taylor Swift的YouTube视频ID
        "nfWlot6h_JM",
        "e-ORhEE9VVg",
        "VuNIsY6JdUw",
        "WA4iX5D9Z64",
        "QcIy9NiNbmo",
        "-CmadmM5cOk",
        "igIfiqqVHtA",
        "IdneKLhsWOQ"
    ],
    "sabrina carpenter": [
        # 请在这里添加Sabrina Carpenter的YouTube视频ID
        "aSugSGCC12I",
        "eVli-tstM5E",
        "cF1Na4AIecM",
        "KEG7b851Ric"
    ],
    "billie eilish": [
        # 请在这里添加Billie Eilish的YouTube视频ID
        "V9PVRfjEBT",
        "BY_XwvKogC8",
        "l08Zw-RY__Q",
        "MB3VkzPdgLA",
        "viimfQi_pUw",
        "pbMwTqkKSps",
        "-tn2S3kJlyU",
        "DyDfgMOUjCI"
    ],
    "ariana grande": [
        # 请在这里添加Ariana Grande的YouTube视频ID
        "tcYodQoapMg",
        "KNtJGQkC-WI",
        "gl1aHhXnN1k",
        "QYh6mYIJG2Y"
        "SXiSVQZLje8",
        "Wg92RrNhB8s",
        "9WbCfHutDSE",
        "_sV0S8qWSy0",
        "iS1g8G_njx8"
        
    ],
    "olivia rodrigo": [
        # 请在这里添加Olivia Rodrigo的YouTube视频ID
        "cii6ruuycQA",
        "ZmDBbnmKpqQ",
        "CRrf3h9vhp8",
        "ZQFmRXgeR-s",
        "RlPNh_PBZb4",
        "gNi_6U5Pm_o"
    ],
    "the weeknd": [
        "XXYlFuWEuKI",  # Blinding Lights
        "4NRXx6U8ABQ",  # Save Your Tears
        "yzTuBuRdAyA",  # Starboy
        "qFLhGq0060w",  # The Hills
    ],
    "bruno mars": [
        "SR6iYWJxHqs",  # That's What I Like
        "OPf0YbXqDm0",  # Uptown Funk
        "e-fA-gBCkj0",  # 24K Magic
        "C9cP9jJkqyA",  # When I Was Your Man
        "nPvuNsRccVw",  # Just The Way You Are
    ],

    "harry styles": [
        "E07s5ZYygMg",  # As It Was
        "9NZvM1918_E",  # Watermelon Sugar
        "VF-r5TtlT9w",  # Sign of the Times
        "olGSAVOkkTI",  # Adore You
    ],

    "bad bunny": [
        "OSUxrSe5GbI",  # Tití Me Preguntó
        "saGYMhApaH8",  # MONACO
        "doLMt10ytHY",  # Me Porto Bonito
        "KU5V5WZVcVE",  # Dakiti
    ],

    "lady gaga": [
        "vBynw9Isr28",  # Bad Romance
        "qrO4YZeyl0I",  # Poker Face
        "bESGLojNYSo",  # Shallow (with Bradley Cooper)
        "EVBsypHzF3U",  # Stupid Love
    ],

    "kendrick lamar": [
        "H58vbez_m4E",  # HUMBLE.
        "U8F5G5wR1mk",  # DNA.
        "tvTRZJ-4EyI",  # Alright
        "GF8aaTu2kg0",  # LOVE. ft. Zacari
    ],

    "sza": [
        "SQnc1QibapQ",  # Kill Bill
        "Sv5yCzPCkv8",  # Snooze
        "0Exxu8lsGYE",  # Good Days
        "Oxf8ULSB8yU",  # Nobody Gets Me
    ],

    "chappell roan": [
        "woLfAvD5iXI",  # Red Wine Supernova
        "GR3Liudev18",  # Hot To Go!
        "ePsqyPMIg6I",  # Casual
    ],

    "teddy swims": [
        "JX7GGkTDAIg",  # Lose Control
        "Qh8QwVYOSVU",
        "9gWIIIr2Asw",
        "VSXT4a2kRHA",
        "_QWZQh0YYWA"
    ],

    "morgan wallen": [
        "uKmg4UbnDLo",  # Wasted On You
        "bUjPPBxbQrQ",  # Last Night
        "X4r_eTtS4Lo",  # Thought You Should Know
        "FXzE9eP1U_E",  # Sand In My Boots
    ],
}

# ------------------------ 全局并发控制 ------------------------
YTDLP_SEMA: Optional[threading.BoundedSemaphore] = None

# ------------------------ 工具函数 ------------------------
def slugify(text: str) -> str:
    text = re.sub(r"[\\/:*?\"<>|]+", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:140] if text else "unknown"

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def ffmpeg_to_wav(src: pathlib.Path, dst: pathlib.Path, sr: int = 16000) -> bool:
    try:
        subprocess.run(
            ["ffmpeg","-y","-i",str(src),"-ac","1","-ar",str(sr),str(dst)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return True
    except Exception:
        return False

def download_stream(url: str, path: pathlib.Path, headers: Dict = UA, timeout: int = 120) -> bool:
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
            if r.status_code == 404: return False
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for ch in r.iter_content(1<<20):
                    if ch: f.write(ch)
        tmp.rename(path); return True
    except Exception:
        if tmp.exists(): tmp.unlink(missing_ok=True)
        return False

# ------------------------ 照片获取（保持原有逻辑） ------------------------
def commons_search_images(name: str, chinese_name: str = "", limit: int = 20) -> List[Dict]:
    search_terms = [name] + ([chinese_name] if chinese_name else [])
    all_results = []
    for search_name in search_terms:
        search = f'intitle:"{search_name}" profile:images ' + " ".join(IMG_KEYWORDS)
        try:
            r = requests.get(COMMONS_API, params={
                "action": "query", "format": "json", "prop": "imageinfo", "generator": "search",
                "gsrsearch": search, "gsrlimit": str(min(limit, 50)),
                "iiprop": "url|extmetadata", "iiurlwidth": "1600",
            }, headers=UA, timeout=60)
            r.raise_for_status()
            pages = (r.json().get("query", {}) or {}).get("pages", {})
            for it in pages.values():
                ii = (it.get("imageinfo") or [{}])[0]
                meta = ii.get("extmetadata") or {}
                lic = (meta.get("LicenseShortName", {}).get("value", "") or "").lower()
                if not any(k in lic for k in ALLOW_LICENSE):
                    continue
                all_results.append({
                    "title": it.get("title"),
                    "thumburl": ii.get("thumburl") or ii.get("url"),
                    "license": lic,
                    "artist": meta.get("Artist", {}).get("value", ""),
                    "credit": meta.get("Credit", {}).get("value", ""),
                    "source": ii.get("descriptionurl"),
                    "search_term": search_name
                })
        except Exception as e:
            print(f"    [照片][ERR] Commons 搜索 '{search_name}' 失败: {e}")
            continue

    # 去重并限制数量
    seen = set(); unique_results = []
    for item in all_results:
        url = item.get("thumburl")
        if url and url not in seen:
            seen.add(url)
            unique_results.append(item)
            if len(unique_results) >= limit:
                break
    return unique_results

def wikipedia_lead_image(name: str) -> Optional[str]:
    import urllib.parse
    t = urllib.parse.quote(name.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{t}"
    try:
        r = requests.get(url, headers=UA, timeout=20)
        if r.status_code == 404: return None
        r.raise_for_status()
        js = r.json()
        return (js.get("thumbnail") or {}).get("source")
    except Exception:
        return None

def fetch_images_for_artist(artist_info: Dict, out_dir: pathlib.Path, max_images: int = 10) -> pd.DataFrame:
    ensure_dir(out_dir)
    name = artist_info["name"]; chinese_name = artist_info.get("chinese_name", "")
    rows = commons_search_images(name, chinese_name, limit=max_images*2)
    if len(rows) < 3:
        for search_name in [name, chinese_name] if chinese_name else [name]:
            lead = wikipedia_lead_image(search_name)
            if lead:
                rows.append({
                    "title": f"wikipedia_lead_{search_name}", 
                    "thumburl": lead, 
                    "license": "unknown", 
                    "artist": "", 
                    "credit": "", 
                    "source": "wikipedia",
                    "search_term": search_name
                })
                break
    saved = []; seen = set()
    for i, im in enumerate(rows, 1):
        url = im.get("thumburl")
        if not url or url in seen: continue
        seen.add(url)
        ext = ".jpg"
        if url.lower().endswith((".png", ".webp", ".gif")):
            ext = "." + url.split(".")[-1].lower()
        path = out_dir / f"{i:04d}{ext}"
        ok = download_stream(url, path)
        if ok: saved.append({"file": str(path), **im})
        if len(saved) >= max_images: break
        time.sleep(0.3)
    return pd.DataFrame(saved)

# ------------------------ 手动YouTube音频下载 ------------------------
def yt_dlp_download_audio_by_id(video_id: str, out_dir: pathlib.Path, sr: int = 16000,
                                 cookies: Optional[str] = None,
                                 timeout: int = 600) -> Optional[pathlib.Path]:
    """
    用 yt-dlp 抓取指定 YouTube 视频ID 的音频，转 wav（单声道、sr=16000）。
    成功/失败都会 print，失败会写日志文件。
    """
    ensure_dir(out_dir)
    
    # 构建 YouTube URL
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # 检查是否已经下载过
    existing_wav = out_dir / f"{video_id}.wav"
    if existing_wav.exists():
        print(f"        [抓取][SKIP] 已存在: {existing_wav.name}")
        return existing_wav
    
    tmpl = str(out_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "-o", tmpl,
        "--no-playlist",
        "--no-warnings",
        "--ignore-errors",
        "--force-ipv4",
        "--socket-timeout", "20",
        "--retries", "3",
        "--extractor-retries", "3",
    ]
    if cookies:
        if cookies.startswith("from_browser:"):
            cmd += ["--cookies-from-browser", cookies.split(":",1)[1]]
        else:
            cmd += ["--cookies", cookies]

    logf = out_dir / f"yt_dlp_{video_id}_{int(time.time())}.log"
    try:
        # 控制 yt-dlp 并发
        if YTDLP_SEMA is not None:
            YTDLP_SEMA.acquire()
        try:
            with open(logf, "wb") as lf:
                print(f"        [抓取] yt-dlp 开始: {video_id}")
                proc = subprocess.run(cmd + [url], stdout=lf, stderr=lf, timeout=timeout)
        finally:
            if YTDLP_SEMA is not None:
                YTDLP_SEMA.release()

        if proc.returncode != 0:
            print(f"        [抓取][ERR] yt-dlp 失败 (ID={video_id})，日志: {logf}")
            return None
        else:
            print(f"        [抓取][OK ] yt-dlp 成功 (ID={video_id})，过程日志: {logf}")

        # 找到刚下载的源文件
        downloaded_file = None
        for p in out_dir.glob(f"{video_id}.*"):
            if p.suffix.lower() in {".m4a",".webm",".opus",".mp3",".wav",".flac"}:
                downloaded_file = p
                break
        
        if downloaded_file is None:
            print(f"        [抓取][ERR] 未找到输出文件 (ID={video_id})")
            return None

        wav = out_dir / f"{video_id}.wav"
        if have_ffmpeg() and downloaded_file.suffix.lower() != ".wav":
            ok = ffmpeg_to_wav(downloaded_file, wav, sr=sr)
            if not ok:
                print(f"        [转码][ERR] ffmpeg 转 wav 失败 (ID={video_id})")
                return None
            else:
                print(f"        [转码][OK ] -> {wav.name}")
                # 删除原始文件以节省空间
                downloaded_file.unlink(missing_ok=True)
        else:
            if downloaded_file.suffix.lower() != ".wav":
                print(f"        [转码][WARN] 未检测到 ffmpeg，保留原始格式 (ID={video_id})")
            wav = downloaded_file

        return wav
    except subprocess.TimeoutExpired:
        print(f"        [抓取][ERR] 超时 (>{timeout}s) (ID={video_id})，日志: {logf}")
        return None
    except Exception as e:
        print(f"        [抓取][EXC] {e} (ID={video_id})，日志: {logf}")
        return None

def fetch_manual_youtube_audio(artist_info: Dict, out_dir: pathlib.Path,
                               cookies: Optional[str] = None) -> pd.DataFrame:
    """
    根据手动指定的YouTube视频ID下载音频
    """
    rows = []
    ensure_dir(out_dir)
    
    artist_name = artist_info["name"].lower()
    video_ids = MANUAL_YOUTUBE_IDS.get(artist_name, [])
    
    if not video_ids:
        print(f"    [音频][WARN] 未为 '{artist_info['name']}' 指定任何YouTube视频ID")
        return pd.DataFrame(rows)
    
    print(f"    [音频] 开始下载手动指定的 {len(video_ids)} 个视频...")
    
    for i, video_id in enumerate(video_ids, 1):
        print(f"    [音频] 处理 {i}/{len(video_ids)}: {video_id}")
        wav = yt_dlp_download_audio_by_id(video_id, out_dir, cookies=cookies)
        
        if wav and wav.exists():
            rows.append({
                "file": str(wav),
                "source": "YouTube",
                "video_id": video_id,
                "identifier": video_id,
                "license": "all-rights-reserved",
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "manually_selected": True
            })
            print(f"        [音频][OK ] 成功下载: {video_id}")
        else:
            print(f"        [音频][FAIL] 下载失败: {video_id}")
        
        time.sleep(0.5)  # 避免请求过快

    print(f"    [音频] 完成，成功下载 {len(rows)}/{len(video_ids)} 个音频文件")
    return pd.DataFrame(rows)

# ------------------------ 单歌手处理 ------------------------
def process_one_artist(artist_info: Dict,
                       out_root: pathlib.Path,
                       images_per_artist: int,
                       yt_cookies: Optional[str]) -> Dict:
    """
    并行工作单元：处理一个歌手，返回结果字典
    """
    name = artist_info["name"]
    chinese_name = artist_info.get("chinese_name", "")
    region = artist_info.get("region", "")
    genre = artist_info.get("genre", "")

    result = {"name": name, "ok": True, "audio_ok": False, "images_ok": False, "err": None}

    try:
        print(f"\n=== {name} ===")
        if chinese_name:
            print(f"    中文名: {chinese_name}")
        print(f"    地区: {region}, 类型: {genre}")

        slug = slugify(name)
        base = out_root / slug
        img_dir = base / "images"
        aud_dir = base / "audio"
        ensure_dir(base)

        # 元信息
        meta = {
            "name": name,
            "chinese_name": chinese_name,
            "region": region,
            "genre": genre,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        (base / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

        # -------- 照片部分（如已存在则跳过）--------
        img_csv = base / "images.csv"
        has_images = False
        if img_csv.exists():
            try:
                df_existing = pd.read_csv(img_csv)
                has_images = not df_existing.empty and len(df_existing) > 0
                if has_images:
                    print(f"    [照片] 跳过（已存在 {len(df_existing)} 张图片）")
            except:
                has_images = False
        
        # 如果CSV不存在或为空，检查目录中是否有图片文件
        if not has_images and img_dir.exists():
            image_files = list(img_dir.glob("*.*"))
            image_files = [f for f in image_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.gif'}]
            if image_files:
                has_images = True
                print(f"    [照片] 跳过（目录中已存在 {len(image_files)} 个图片文件）")

        if not has_images:
            print("    [照片] 搜索中...")
            df_img = fetch_images_for_artist(artist_info, img_dir, max_images=images_per_artist)
            if not df_img.empty:
                df_img.to_csv(img_csv, index=False, encoding='utf-8-sig')
                print(f"    [照片] 成功保存 {len(df_img)} 张")
                result["images_ok"] = True
            else:
                print("    [照片][WARN] 未找到可用图片")
        else:
            result["images_ok"] = True

        # -------- 音频部分（手动指定YouTube ID）--------
        print("    [音频] 手动YouTube ID抓取中...")
        aud_csv = base / "audio.csv"

        df_aud = fetch_manual_youtube_audio(artist_info, aud_dir, cookies=yt_cookies)
        if not df_aud.empty:
            df_aud.to_csv(aud_csv, index=False, encoding='utf-8-sig')
            print(f"    [音频] 成功抓取 {len(df_aud)} 段音频 -> {aud_csv.name}")
            result["audio_ok"] = True
        else:
            print("    [音频][WARN] 未成功抓取任何音频（检查MANUAL_YOUTUBE_IDS配置）。")

    except Exception as e:
        result["ok"] = False
        result["err"] = f"{type(e).__name__}: {e}"
        print(f"[FATAL] {name} 处理失败：{result['err']}")
        traceback.print_exc()

    return result

# ------------------------ 主流程（并行） ------------------------
def build_singer_dataset_parallel(out_root: pathlib.Path,
                                  images_per_artist: int,
                                  artists_filter: Optional[List[str]],
                                  yt_cookies: Optional[str],
                                  max_workers: int) -> None:
    ensure_dir(out_root)

    # 过滤歌手
    singers = POPULAR_SINGERS_2024_2025
    if artists_filter:
        low = [af.lower() for af in artists_filter]
        singers = [s for s in singers if s["name"].lower() in low]
    if not singers:
        print("[FATAL] 没有找到匹配的歌手")
        return

    # 检查哪些歌手有手动指定的视频ID
    singers_with_ids = []
    singers_without_ids = []
    for s in singers:
        artist_key = s["name"].lower()
        if artist_key in MANUAL_YOUTUBE_IDS and MANUAL_YOUTUBE_IDS[artist_key]:
            singers_with_ids.append(s)
        else:
            singers_without_ids.append(s)
    
    print(f"=== YouTube ID 配置检查 ===")
    print(f"已配置音频ID的歌手: {len(singers_with_ids)} 位")
    for s in singers_with_ids:
        ids_count = len(MANUAL_YOUTUBE_IDS[s["name"].lower()])
        print(f"  - {s['name']}: {ids_count} 个视频ID")
    
    if singers_without_ids:
        print(f"未配置音频ID的歌手: {len(singers_without_ids)} 位")
        for s in singers_without_ids:
            print(f"  - {s['name']} (将只下载图片)")

    # 保存歌手列表（只写一次）
    singers_df = pd.DataFrame(singers)
    seed_csv = out_root / "popular_singers_2024_2025.csv"
    singers_df.to_csv(seed_csv, index=False, encoding='utf-8-sig')
    print(f"\n将构建数据集，包含 {len(singers)} 位流行歌手 -> {seed_csv}")

    results = []
    # 并行执行
    with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        jobs = []
        for s in singers:
            jobs.append(ex.submit(
                process_one_artist, s, out_root, images_per_artist, yt_cookies
            ))
        for f in futures.as_completed(jobs):
            results.append(f.result())

    # 汇总
    ok_cnt = sum(1 for r in results if r.get("ok"))
    img_ok = sum(1 for r in results if r.get("images_ok"))
    aud_ok = sum(1 for r in results if r.get("audio_ok"))
    print("\n=== 完成 ===")
    print(f"成功处理 {ok_cnt}/{len(results)} 位歌手")
    print(f"  有图片的歌手：{img_ok}")
    print(f"  有音频的歌手：{aud_ok}")
    print("数据保存位置:", out_root.absolute())
    print("文件说明:")
    print("  - popular_singers_2024_2025.csv: 歌手基本信息")
    print("  - [歌手目录]/meta.json: 歌手详细信息")
    print("  - [歌手目录]/images.csv: 图片信息")
    print("  - [歌手目录]/audio.csv: 手动指定的YouTube音频（wav）")

# ------------------------ 预检 ------------------------
def _preflight() -> bool:
    ok = True
    from shutil import which
    if which("yt-dlp") is None:
        print("[FATAL] 未找到 yt-dlp：请安装 `pip install yt-dlp` 或加载模块，并确保在运行节点可用。")
        ok = False
    if which("ffmpeg") is None:
        print("[WARN] 未找到 ffmpeg：将不会转码为 wav（仍可使用原始 m4a/webm），建议安装。")
    
    # 检查手动配置的YouTube ID数量
    total_ids = sum(len(ids) for ids in MANUAL_YOUTUBE_IDS.values() if ids)
    configured_artists = sum(1 for ids in MANUAL_YOUTUBE_IDS.values() if ids)
    print(f"[INFO] 手动配置状态：{configured_artists} 位歌手，共 {total_ids} 个YouTube视频ID")
    
    if total_ids == 0:
        print("[WARN] MANUAL_YOUTUBE_IDS 中没有配置任何YouTube视频ID，将只下载图片。")
    
    return ok

def show_youtube_id_template():
    """显示YouTube ID配置模板，方便用户填写"""
    print("\n=== YouTube ID 配置模板 ===")
    print("请在 MANUAL_YOUTUBE_IDS 字典中为每位歌手添加YouTube视频ID：")
    print()
    for singer in POPULAR_SINGERS_2024_2025:
        key = singer["name"].lower()
        print(f'    "{key}": [')
        print(f'        # 请添加{singer["name"]}的YouTube视频ID，例如：')
        print('        # "dQw4w9WgXcQ",  # 歌曲名称1')
        print('        # "abc123def45",  # 歌曲名称2')
        print('    ],')
    print()
    print("YouTube视频ID获取方法：")
    print("1. 打开YouTube视频页面")
    print("2. 从URL中复制 'watch?v=' 后面的ID部分")
    print("3. 例如：https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print("   对应的ID就是：dQw4w9WgXcQ")

# ------------------------ 入口 ------------------------
def main():
    global YTDLP_SEMA

    ap = argparse.ArgumentParser(description="构建2024-2025热门流行歌手数据集（图片 + 手动指定YouTube音频，无切分，带日志，并行）")
    ap.add_argument("--out", default="pop_singers_manual_yt_2024_2025", type=str, help="输出根目录")
    ap.add_argument("--images-per-artist", default=8, type=int, help="每位歌手最多图片数")
    ap.add_argument("--artists", nargs="+", help="只处理指定歌手（英文名，空格分隔）")
    ap.add_argument("--list-artists", action="store_true", help="列出所有支持的歌手")
    ap.add_argument("--show-template", action="store_true", help="显示YouTube ID配置模板")

    ap.add_argument("--yt-cookies", type=str, help="cookies.txt 路径，或 'from_browser:chrome' 等（优先于默认路径）")

    ap.add_argument("--max-workers", type=int, default=6, help="并行处理歌手的线程数")
    ap.add_argument("--yt-dlp-parallel", type=int, default=3, help="同时进行的 yt-dlp 抓取数上限（避免被限流）")

    args = ap.parse_args()

    if args.list_artists:
        print("=== 支持的流行歌手列表（2024-2025） ===")
        print("\n华语歌手:")
        for s in POPULAR_SINGERS_2024_2025:
            if s.get("chinese_name"):
                key = s["name"].lower()
                ids_count = len(MANUAL_YOUTUBE_IDS.get(key, []))
                status = f"({ids_count} IDs)" if ids_count > 0 else "(未配置)"
                print(f"  {s['name']} ({s['chinese_name']}) - {s['region']} {status}")
        print("\n欧美歌手:")
        for s in POPULAR_SINGERS_2024_2025:
            if not s.get("chinese_name"):
                key = s["name"].lower()
                ids_count = len(MANUAL_YOUTUBE_IDS.get(key, []))
                status = f"({ids_count} IDs)" if ids_count > 0 else "(未配置)"
                print(f"  {s['name']} - {s['region']} {status}")
        print(f"\n总计: {len(POPULAR_SINGERS_2024_2025)} 位歌手")
        return

    if args.show_template:
        show_youtube_id_template()
        return

    if not _preflight():
        return

    # cookies 优先使用命令行；否则按你的本地约定复制一份到 /scratch
    yt_cookies = args.yt_cookies
    if not yt_cookies:
        src = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/solos/cookies.txt"
        dst = "/scratch/xwang378/2025/AudioBench/benchmark/Data/solos/cookies_copy.txt"
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.copy(src, dst)
                yt_cookies = dst
                print(f"[INFO] 已复制 cookies: {src} -> {dst}")
            except Exception as e:
                print(f"[WARN] cookies 复制失败：{e}")
        else:
            # 也可以支持 from_browser
            if os.getenv("YT_COOKIES", ""):
                yt_cookies = os.getenv("YT_COOKIES")
                print(f"[INFO] 使用环境变量 YT_COOKIES: {yt_cookies}")

    # 初始化 yt-dlp 并发信号量
    YTDLP_SEMA = threading.BoundedSemaphore(value=max(1, args.yt_dlp_parallel))

    out_root = pathlib.Path(args.out)

    build_singer_dataset_parallel(
        out_root=out_root,
        images_per_artist=args.images_per_artist,
        artists_filter=args.artists,
        yt_cookies=yt_cookies,
        max_workers=max(1, args.max_workers),
    )

if __name__ == "__main__":
    main()