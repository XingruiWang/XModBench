#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 2024-2025 流行歌手数据集（图片 + YouTube 音频，带详细日志 & 并行）
- 图片：Wikimedia Commons（开放许可） + Wikipedia 首图
- 音频：YouTube（yt-dlp 抓取 bestaudio -> ffmpeg 转 wav，失败有日志与 print）
- 候选来源：--seed-youtube 种子（CSV: artist,url） + 可选 YouTube Data API 搜索（需 YT_API_KEY）
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

# ------------------------ 照片获取 ------------------------
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

# ------------------------ YouTube 搜索 & 抓取 ------------------------
def yt_search_music_videos(artist_name: str, chinese_name: str = "", max_results: int = 8) -> List[Dict]:
    """
    用 YouTube Data API 搜索候选；无 YT_API_KEY 时返回空
    """
    key = os.getenv("YT_API_KEY")
    if not key:
        return []
    search_names = [artist_name] + ([chinese_name] if chinese_name else [])
    all_results = []; seen_ids = set()
    for name in search_names:
        q = f'{name} official music video OR lyric video OR live performance'
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": q,
            "type": "video",
            "maxResults": max_results,
            "videoCategoryId": "10",
            "safeSearch": "none",
            "key": key,
        }
        try:
            r = requests.get(url, params=params, headers=UA, timeout=20)
            r.raise_for_status()
            for it in r.json().get("items", []):
                vid = it["id"]["videoId"]
                if vid in seen_ids: continue
                seen_ids.add(vid)
                sn = it["snippet"]
                all_results.append({
                    "video_id": vid,
                    "title": sn.get("title",""),
                    "channel": sn.get("channelTitle",""),
                    "published_at": sn.get("publishedAt",""),
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "search_term": name
                })
        except Exception as e:
            print(f"    [音频][ERR] YouTube 搜索 '{name}' 失败: {e}")
            continue
    return all_results

def yt_dlp_download_audio(url: str, out_dir: pathlib.Path, sr: int = 16000,
                          cookies: Optional[str] = None,
                          timeout: int = 600) -> Optional[pathlib.Path]:
    """
    用 yt-dlp 抓取单条 YouTube 音频，转 wav（单声道、sr=16000）。
    成功/失败都会 print，失败会写日志文件。
    cookies: cookies.txt 路径，或 'from_browser:chrome'/'from_browser:chromium' 等
    """
    ensure_dir(out_dir)
    import ipdb; ipdb.set_trace()
    tmpl = str(out_dir / "%(id)s.%(ext)s")
    if os.path.exists(tmpl):
        print(f"        [抓取][WARN] 输出目录已存在: {out_dir}")
        return None
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

    logf = out_dir / (f"yt_dlp_{int(time.time())}.log")
    try:
        # 控制 yt-dlp 并发
        if YTDLP_SEMA is not None:
            YTDLP_SEMA.acquire()
        try:
            with open(logf, "wb") as lf:
                print(f"        [抓取] yt-dlp 开始: {url}")
                proc = subprocess.run(cmd + [url], stdout=lf, stderr=lf, timeout=timeout)
        finally:
            if YTDLP_SEMA is not None:
                YTDLP_SEMA.release()

        if proc.returncode != 0:
            print(f"        [抓取][ERR] yt-dlp 失败 (url={url})，日志: {logf}")
            return None
        else:
            print(f"        [抓取][OK ] yt-dlp 成功 (url={url})，过程日志: {logf}")

        # 找到刚下载的源文件
        newest = None
        for p in out_dir.glob("*.*"):
            if p.suffix.lower() in {".m4a",".webm",".opus",".mp3",".wav",".flac"}:
                if newest is None or p.stat().st_mtime > newest.stat().st_mtime:
                    newest = p
        if newest is None:
            print("        [抓取][ERR] 未找到输出文件")
            return None

        wav = out_dir / (newest.stem + ".wav")
        if have_ffmpeg():
            ok = ffmpeg_to_wav(newest, wav, sr=sr)
            if not ok:
                print("        [转码][ERR] ffmpeg 转 wav 失败")
                return None
            else:
                print(f"        [转码][OK ] -> {wav.name}")
        else:
            print("        [转码][WARN] 未检测到 ffmpeg，保留原始格式作为输出")
            wav = newest

        return wav
    except subprocess.TimeoutExpired:
        print(f"        [抓取][ERR] 超时 (>{timeout}s) (url={url})，日志: {logf}")
        return None
    except Exception as e:
        print(f"        [抓取][EXC] {e} (url={url})，日志: {logf}")
        return None

def fetch_audio_from_youtube_candidates(artist_info: Dict, out_dir: pathlib.Path,
                                        candidates: List[Dict],
                                        want: int = 4,
                                        cookies: Optional[str] = None) -> pd.DataFrame:
    """
    输入：YouTube 候选（来自 yt_search_music_videos 或 --seed-youtube）
    输出：成功下载并转 wav 的若干条记录（失败会 print 提示）
    """
    rows = []
    ensure_dir(out_dir)
    kept = 0
    seen_vid = set()

    def score(x: Dict) -> int:
        title = (x.get("title") or "").lower()
        ch    = (x.get("channel") or "").lower()
        s = 0
        if "official" in title or "mv" in title or "官方" in title:
            s += 3
        if "lyric" in title or "歌词" in title:
            s += 1
        if artist_info["name"].lower() in ch:
            s += 2
        if "live" in title or "现场" in title:
            s += 1
        return s

    # 标准化
    normed = []
    for it in candidates:
        if isinstance(it, dict):
            normed.append(it)
        else:
            normed.append({"video_id": None, "url": str(it), "title": "", "channel": "", "published_at":"", "search_term": artist_info["name"]})

    normed = sorted(normed, key=score, reverse=True)

    print(f"    [音频] 开始下载候选（目标 {want} 条）...")
    for it in normed:
        if kept >= want: break
        vid = it.get("video_id")
        if vid and vid in seen_vid: continue
        if vid: seen_vid.add(vid)
        url = it.get("url") or (f"https://www.youtube.com/watch?v={vid}" if vid else None)
        if not url:
            print("        [候选][WARN] 缺少 url，跳过")
            continue

        wav = yt_dlp_download_audio(url, out_dir=out_dir, sr=16000, cookies=cookies)
        if not wav or not wav.exists():
            print(f"        [候选][FAIL] 下载失败：{url}")
            continue

        rows.append({
            "file": str(wav),
            "source": "YouTube",
            "identifier": vid if vid else pathlib.Path(str(wav)).stem,
            "license": "all-rights-reserved",
            "title": it.get("title",""),
            "channel": it.get("channel",""),
            "published_at": it.get("published_at",""),
            "url": url,
            "search_term": it.get("search_term", artist_info["name"])
        })
        kept += 1
        print(f"        [候选][OK ] 已抓取 {kept}/{want}")
        time.sleep(0.2)

    if kept == 0:
        print(f"    [音频][WARN] 候选 {len(normed)} 个，但全部下载失败（考虑提供 --yt-cookies）。")

    return pd.DataFrame(rows)

# ------------------------ 单歌手处理 ------------------------
def process_one_artist(artist_info: Dict,
                       out_root: pathlib.Path,
                       images_per_artist: int,
                       youtube_max: int,
                       seed_youtube_map: Optional[Dict[str, List[str]]],
                       yt_cookies: Optional[str],
                       use_youtube_search: bool) -> Dict:
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

        # -------- 照片 --------
        img_csv = base / "images.csv"
        has_images = False
        if img_csv.exists():
            try:
                has_images = not pd.read_csv(img_csv).empty
            except:
                has_images = False
        if not has_images and img_dir.exists():
            has_images = any(img_dir.glob("*.*"))

        if has_images:
            print("    [照片] 跳过（已存在）")
            result["images_ok"] = True
        else:
            print("    [照片] 搜索中...")
            df_img = fetch_images_for_artist(artist_info, img_dir, max_images=images_per_artist)
            if not df_img.empty:
                df_img.to_csv(img_csv, index=False, encoding='utf-8-sig')
                print(f"    [照片] 成功保存 {len(df_img)} 张")
                result["images_ok"] = True
            else:
                print("    [照片][WARN] 未找到可用图片")

        # -------- 音频（仅 YouTube）--------
        print("    [音频] YouTube 抓取中...")
        aud_csv = base / "audio.csv"

        # 1) 种子 URL
        seed_list = seed_youtube_map.get(name.lower(), []) if seed_youtube_map else []
        seed_candidates = [{"url": u, "title": "", "channel": "", "published_at": "", "search_term": name} for u in seed_list]

        # 2) YouTube API 搜索（可选）
        search_candidates = yt_search_music_videos(name, chinese_name, max_results=max(8, youtube_max*2)) if use_youtube_search else []

        candidates = seed_candidates + search_candidates

        print(f"    [音频] 候选数量 = {len(candidates)} (seed={len(seed_candidates)}, search={len(search_candidates)})")
        if not candidates:
            if not os.getenv("YT_API_KEY") and not seed_list:
                print("    [音频][WARN] 无候选：未提供 --seed-youtube 且未设置 YT_API_KEY。")
            elif os.getenv("YT_API_KEY") and not search_candidates:
                print("    [音频][WARN] 已设置 YT_API_KEY 但搜索返回 0；可能网络/配额/地域限制。")

        if candidates:
            df_ytaud = fetch_audio_from_youtube_candidates(
                artist_info, aud_dir, candidates, want=youtube_max, cookies=yt_cookies
            )
            if not df_ytaud.empty:
                df_ytaud.to_csv(aud_csv, index=False, encoding='utf-8-sig')
                print(f"    [音频] YouTube 成功抓取 {len(df_ytaud)} 段音频 -> {aud_csv.name}")
                result["audio_ok"] = True
            else:
                print("    [音频][WARN] 本歌手未成功抓取任何音频。")

        # 保存搜索候选（便于溯源）
        if search_candidates:
            try:
                pd.DataFrame(search_candidates).to_csv(base / "youtube.csv", index=False, encoding='utf-8-sig')
                print("    [音频] 已保存搜索候选 -> youtube.csv")
            except Exception as e:
                print(f"    [音频][ERR] 写入 youtube.csv 失败: {e}")

    except Exception as e:
        result["ok"] = False
        result["err"] = f"{type(e).__name__}: {e}"
        print(f"[FATAL] {name} 处理失败：{result['err']}")
        traceback.print_exc()

    return result

# ------------------------ 主流程（并行） ------------------------
def build_singer_dataset_parallel(out_root: pathlib.Path,
                                  images_per_artist: int,
                                  youtube_max: int,
                                  artists_filter: Optional[List[str]],
                                  seed_youtube_map: Optional[Dict[str, List[str]]],
                                  yt_cookies: Optional[str],
                                  use_youtube_search: bool,
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

    # 保存歌手列表（只写一次）
    singers_df = pd.DataFrame(singers)
    seed_csv = out_root / "popular_singers_2024_2025.csv"
    singers_df.to_csv(seed_csv, index=False, encoding='utf-8-sig')
    print(f"将构建数据集，包含 {len(singers)} 位流行歌手 -> {seed_csv}")

    results = []
    # 并行执行
    with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        jobs = []
        for s in singers:
            jobs.append(ex.submit(
                process_one_artist, s, out_root, images_per_artist, youtube_max,
                seed_youtube_map, yt_cookies, use_youtube_search
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
    print("  - [歌手目录]/audio.csv: 已抓取的 YouTube 音频（wav）")
    print("  - [歌手目录]/youtube.csv: YouTube 搜索候选（如启用）")

# ------------------------ 预检 ------------------------
def _preflight() -> bool:
    ok = True
    from shutil import which
    if which("yt-dlp") is None:
        print("[FATAL] 未找到 yt-dlp：请安装 `pip install yt-dlp` 或加载模块，并确保在运行节点可用。")
        ok = False
    if which("ffmpeg") is None:
        print("[WARN] 未找到 ffmpeg：将不会转码为 wav（仍可使用原始 m4a/webm），建议安装。")
    if os.getenv("YT_API_KEY") in (None, "", "None"):
        print("[INFO] 未设置 YT_API_KEY：仅会使用 --seed-youtube 的种子链接；若未提供 seeds，将无搜索候选。")
    return ok

# ------------------------ 入口 ------------------------
def main():
    global YTDLP_SEMA

    ap = argparse.ArgumentParser(description="构建2024-2025热门流行歌手数据集（图片 + YouTube 音频，无切分，带日志，并行）")
    ap.add_argument("--out", default="pop_singers_yt_2024_2025", type=str, help="输出根目录")
    ap.add_argument("--images-per-artist", default=8, type=int, help="每位歌手最多图片数")
    ap.add_argument("--youtube-max", default=4, type=int, help="每位歌手最多下载多少条 YouTube 音频")
    ap.add_argument("--artists", nargs="+", help="只处理指定歌手（英文名，空格分隔）")
    ap.add_argument("--list-artists", action="store_true", help="列出所有支持的歌手")

    ap.add_argument("--seed-youtube", type=str, help="CSV with columns: artist,url（手工提供官方MV等种子链接）")
    ap.add_argument("--yt-cookies", type=str, help="cookies.txt 路径，或 'from_browser:chrome' 等（优先于默认路径）")
    ap.add_argument("--no-youtube-search", action="store_true", help="禁用 YouTube Data API 搜索（仅用 --seed-youtube）")

    ap.add_argument("--max-workers", type=int, default=6, help="并行处理歌手的线程数")
    ap.add_argument("--yt-dlp-parallel", type=int, default=3, help="同时进行的 yt-dlp 抓取数上限（避免被限流）")

    args = ap.parse_args()

    if args.list_artists:
        print("=== 支持的流行歌手列表（2024-2025） ===")
        print("\n华语歌手:")
        for s in POPULAR_SINGERS_2024_2025:
            if s.get("chinese_name"):
                print(f"  {s['name']} ({s['chinese_name']}) - {s['region']}")
        print("\n欧美歌手:")
        for s in POPULAR_SINGERS_2024_2025:
            if not s.get("chinese_name"):
                print(f"  {s['name']} - {s['region']}")
        print(f"\n总计: {len(POPULAR_SINGERS_2024_2025)} 位歌手")
        return

    if not _preflight():
        return

    # 读取 seeds（可选）
    seed_map: Dict[str, List[str]] = {}
    if args.seed_youtube and os.path.exists(args.seed_youtube):
        try:
            df_seed = pd.read_csv(args.seed_youtube)
            for _, r in df_seed.iterrows():
                a = str(r.get("artist","")).strip()
                u = str(r.get("url","")).strip()
                if a and u:
                    seed_map.setdefault(a.lower(), []).append(u)
            print(f"[INFO] 已加载种子链接：{sum(len(v) for v in seed_map.values())} 条")
        except Exception as e:
            print(f"[ERR ] 读取种子 CSV 失败：{e}")

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

    use_search = not args.no_youtube_search
    out_root = pathlib.Path(args.out)

    build_singer_dataset_parallel(
        out_root=out_root,
        images_per_artist=args.images_per_artist,
        youtube_max=args.youtube_max,
        artists_filter=args.artists,
        seed_youtube_map=seed_map if seed_map else None,
        yt_cookies=yt_cookies,
        use_youtube_search=use_search,
        max_workers=max(1, args.max_workers),
    )

if __name__ == "__main__":
    main()
