#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build "Popular Singer" dataset (name + multiple photos + multiple voice clips)
Data sources:
- Singer list/avatars: Wikidata (SPARQL) + Wikimedia Commons (open licenses only)
- Audio: Internet Archive (open licenses); optional Jamendo (set JAMENDO_CLIENT_ID)
Note: For research use only, please retain author and license information.
"""

import os, re, json, time, math, argparse, pathlib, subprocess, shutil
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import requests
import pandas as pd
from tqdm import tqdm

# ------------------------ Configuration ------------------------
UA = {"User-Agent": "PopSingerDataset/1.0 (research; contact: example@example.com)"}
SPARQL = "https://query.wikidata.org/sparql"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
IA_ADV = "https://archive.org/advancedsearch.php"
CAA_THUMB = 1200  # Unused (preserved)
ALLOW_LICENSE = (
    "cc-zero","cc0","cc by","cc-by","cc-by-sa","cc-by-nd","cc-by-nc","cc-by-nc-sa","cc-by-nc-nd",
    "public domain","pd"
)
IMG_KEYWORDS = ("portrait","headshot","singer","vocalist","live","concert","press","promotional")

# Thread-safe rate limiting
class RateLimiter:
    def __init__(self, delay=0.2):
        self.delay = delay
        self.lock = threading.Lock()
        self.last_call = 0
    
    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self.last_call = time.time()

rate_limiter = RateLimiter()

# ------------------------ Utility Functions ------------------------
def slugify(text: str) -> str:
    """Convert text to filesystem-safe filename"""
    text = re.sub(r"[\\/:*?\"<>|]+", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:140] if text else "unknown"

def ensure_dir(p: pathlib.Path):
    """Create directory if it doesn't exist"""
    p.mkdir(parents=True, exist_ok=True)

def download_stream(url: str, path: pathlib.Path, headers: Dict = UA, timeout: int = 120) -> bool:
    """Download file with streaming and partial file protection"""
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        rate_limiter.wait()  # Rate limiting
        with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
            if r.status_code == 404:
                return False
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for ch in r.iter_content(1<<20):
                    if ch: f.write(ch)
        tmp.rename(path)
        return True
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False

def have_ffmpeg() -> bool:
    """Check if ffmpeg is available"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def ffmpeg_to_wav(src: pathlib.Path, dst: pathlib.Path, sr: int = 16000) -> bool:
    """Convert audio file to WAV format using ffmpeg"""
    try:
        subprocess.run(
            ["ffmpeg","-y","-i",str(src),"-ac","1","-ar",str(sr),str(dst)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return True
    except Exception:
        return False

# ------------------------ 1) Artist List (Wikidata) ------------------------
def fetch_artists_from_wikidata(max_artists: int = 200) -> pd.DataFrame:
    """Fetch pop singers from Wikidata using SPARQL"""
    # pop singers: occupation singer/pop singer + genre pop music
    Q_POP = "Q37073"
    query = f"""
    SELECT ?artist ?artistLabel ?mbid ?img ?countryLabel WHERE {{
      ?artist wdt:P31 wd:Q5 .
      ?artist wdt:P106 ?occ .
      VALUES ?occ {{ wd:Q177220 wd:Q753110 }}  # singer / pop singer
      ?artist wdt:P136 wd:{Q_POP} .
      OPTIONAL {{ ?artist wdt:P434 ?mbid . }}
      OPTIONAL {{ ?artist wdt:P18 ?img . }}
      OPTIONAL {{ ?artist wdt:P27 ?country . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {max_artists}
    """
    r = requests.get(SPARQL, params={"query": query, "format": "json"}, headers=UA, timeout=60)
    r.raise_for_status()
    rows = []
    for b in r.json()["results"]["bindings"]:
        rows.append({
            "qid": b["artist"]["value"].split("/")[-1],
            "name": b["artistLabel"]["value"],
            "mbid": b.get("mbid", {}).get("value", ""),
            "image": b.get("img", {}).get("value", ""),
            "country": b.get("countryLabel", {}).get("value", ""),
        })
    df = pd.DataFrame(rows).drop_duplicates("qid")
    return df

# ------------------------ 2) Photos (Commons) ------------------------
def commons_search_images(name: str, limit: int = 20) -> List[Dict]:
    """
    Search images on Commons using generator=search + imageinfo for thumbnails and license metadata
    """
    # Search with keywords; intitle takes priority
    search = f'intitle:"{name}" profile:images ' + " ".join(IMG_KEYWORDS)
    r = requests.get(COMMONS_API, params={
        "action": "query", "format": "json", "prop": "imageinfo", "generator": "search",
        "gsrsearch": search, "gsrlimit": str(limit),
        "iiprop": "url|extmetadata", "iiurlwidth": "1600",
    }, headers=UA, timeout=60)
    r.raise_for_status()
    pages = (r.json().get("query", {}) or {}).get("pages", {})
    out = []
    for it in pages.values():
        ii = (it.get("imageinfo") or [{}])[0]
        meta = ii.get("extmetadata") or {}
        lic = (meta.get("LicenseShortName", {}).get("value", "") or "").lower()
        # Some returns like "CC BY-SA 4.0", normalize them

        out.append({
            "title": it.get("title"),
            "thumburl": ii.get("thumburl") or ii.get("url"),
            "license": lic,
            "artist": meta.get("Artist", {}).get("value", ""),
            "credit": meta.get("Credit", {}).get("value", ""),
            "source": ii.get("descriptionurl"),
        })
    return out

def download_single_image(args) -> Optional[Dict]:
    """Download a single image (for parallel processing)"""
    i, im, out_dir, seen_lock, seen_set = args
    url = im.get("thumburl")
    if not url:
        return None
    
    with seen_lock:
        if url in seen_set:
            return None
        seen_set.add(url)
    
    path = out_dir / f"{i:04d}.jpg"
    ok = download_stream(url, path)
    if ok:
        return {"file": str(path), **im}
    return None

def fetch_images_for_artist(artist_name: str, out_dir: pathlib.Path, max_images: int = 10, p18_url: str = "") -> pd.DataFrame:
    """Fetch images for a single artist with parallel downloading"""
    ensure_dir(out_dir)
    rows = []
    seen_set = set()
    seen_lock = threading.Lock()
    
    # First collect P18 (if available)
    if p18_url:
        rows.append({"title":"P18", "thumburl": p18_url, "license":"unknown", "artist":"", "credit":"", "source": p18_url})
    
    # Then search for more
    rows += commons_search_images(artist_name, limit=max_images*2)
    
    # Prepare download tasks
    download_tasks = []
    for i, im in enumerate(rows, 1):
        download_tasks.append((i, im, out_dir, seen_lock, seen_set))
    
    saved = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_task = {executor.submit(download_single_image, task): task for task in download_tasks}
        
        for future in as_completed(future_to_task):
            result = future.result()
            if result:
                saved.append(result)
            if len(saved) >= max_images:
                # Cancel remaining tasks
                for f in future_to_task:
                    f.cancel()
                break
    
    return pd.DataFrame(saved)

# ------------------------ 3) Audio (Internet Archive + Optional Jamendo) ------------------------
def ia_search_items(artist: str, rows: int = 40) -> List[Dict]:
    """Search audio items on Internet Archive"""
    q = f"\"{artist}\" AND mediatype:audio"
    params = {
        "q": q,
        "fl[]": ["identifier","title","licenseurl","creator"],
        "output": "json",
        "rows": str(rows)
    }
    r = requests.get(IA_ADV, params=params, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()["response"]["docs"]

def ia_list_audio_files(identifier: str) -> List[Tuple[str,str]]:
    """List audio files in an Internet Archive item"""
    r = requests.get(f"https://archive.org/metadata/{identifier}", headers=UA, timeout=30)
    r.raise_for_status()
    files = r.json().get("files", [])
    out = []
    for f in files:
        fmt = (f.get("format") or "").lower()
        if fmt.startswith(("mp3","ogg","flac","wav","mpeg")):
            out.append((f.get("name"), fmt))
    return out

def download_and_convert_audio(args) -> Optional[Dict]:
    """Download and convert a single audio file (for parallel processing)"""
    url, tmp_path, wav_path, metadata = args
    
    if not download_stream(url, tmp_path):
        return None
    
    # Convert to WAV
    if have_ffmpeg():
        ok = ffmpeg_to_wav(tmp_path, wav_path, sr=16000)
        tmp_path.unlink(missing_ok=True)
        if not ok:
            return None
    else:
        # No ffmpeg, keep original file
        wav_path = tmp_path
    
    return {
        "file": str(wav_path),
        **metadata
    }

def fetch_audio_for_artist(artist_name: str, out_dir: pathlib.Path, max_clips: int = 6) -> pd.DataFrame:
    """Fetch audio files for a single artist with parallel downloading"""
    ensure_dir(out_dir)
    rows = []
    
    # Internet Archive
    docs = ia_search_items(artist_name, rows=max(40, max_clips*5))
    download_tasks = []
    
    for d in docs:
        lic = (d.get("licenseurl") or "").lower()
        ident = d["identifier"]
        
        try:
            files = ia_list_audio_files(ident)
        except Exception:
            continue
        
        for fname, fmt in files:
            if len(download_tasks) >= max_clips:
                break
                
            url = f"https://archive.org/download/{ident}/{fname}"
            tmp = out_dir / f"{ident}_{slugify(fname)}"
            wav = out_dir / (tmp.stem + ".wav")
            
            metadata = {
                "source": "InternetArchive",
                "identifier": ident,
                "license": lic,
                "title": d.get("title","")
            }
            
            download_tasks.append((url, tmp, wav, metadata))
            
        if len(download_tasks) >= max_clips:
            break
    
    # Execute downloads in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_task = {executor.submit(download_and_convert_audio, task): task for task in download_tasks}
        
        for future in as_completed(future_to_task):
            result = future.result()
            if result:
                rows.append(result)
            if len(rows) >= max_clips:
                # Cancel remaining tasks
                for f in future_to_task:
                    f.cancel()
                break
    
    # Jamendo (optional)
    if len(rows) < max_clips and os.getenv("JAMENDO_CLIENT_ID"):
        cid = os.getenv("JAMENDO_CLIENT_ID")
        url = f"https://api.jamendo.com/v3.0/tracks/?client_id={cid}&format=json&limit={max_clips-len(rows)}&fuzzytags=vocal&search={requests.utils.quote(artist_name)}&audioformat=mp32"
        try:
            r = requests.get(url, headers=UA, timeout=30); r.raise_for_status()
            jamendo_tasks = []
            
            for tr in r.json().get("results", []):
                mp3 = tr["audio"]
                tmp = out_dir / f"jamendo_{tr['id']}.mp3"
                wav = out_dir / (tmp.stem + ".wav")
                
                metadata = {
                    "source": "Jamendo",
                    "identifier": tr["id"],
                    "license": tr.get("licenseCC",""),
                    "title": tr.get("name","")
                }
                
                jamendo_tasks.append((mp3, tmp, wav, metadata))
            
            # Execute Jamendo downloads
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_task = {executor.submit(download_and_convert_audio, task): task for task in jamendo_tasks}
                
                for future in as_completed(future_to_task):
                    result = future.result()
                    if result:
                        rows.append(result)
                    if len(rows) >= max_clips:
                        break
        except Exception:
            pass

    return pd.DataFrame(rows)

# ------------------------ 4) Audio Segmentation ------------------------
def segment_audio_simple(wav_path: pathlib.Path, out_dir: pathlib.Path,
                         min_len: float = 6.0, max_len: float = 12.0,
                         thr: float = 0.02, max_segments: int = 8) -> List[Dict]:
    """
    Simple energy threshold segmentation (no model required); needs librosa/soundfile
    """
    try:
        import librosa, soundfile as sf
    except ImportError:
        print("Warning: librosa/soundfile not available, skipping segmentation")
        return []
    
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    if len(y) == 0: return []
    
    frame_len = int(0.05*sr); hop = int(0.02*sr)
    if len(y) < frame_len: return []
    
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop, axis=0)
    rms = (frames**2).mean(axis=0)**0.5
    mask = (rms > thr).astype(int)

    starts, ends, on = [], [], None
    for i, m in enumerate(mask):
        if m and on is None: on = i
        if (not m) and on is not None:
            starts.append(on); ends.append(i); on = None
    if on is not None: starts.append(on); ends.append(len(mask))

    segs = []
    for s, e in zip(starts, ends):
        t0, t1 = s*hop/sr, e*hop/sr
        dur = t1 - t0
        if dur < min_len: continue
        t = t0
        while t + min_len <= t1 and len(segs) < max_segments:
            tt = min(t + max_len, t1)
            segs.append((t, tt))
            t += max_len
    
    out = []
    for i, (a, b) in enumerate(segs, 1):
        clip = y[int(a*sr):int(b*sr)]
        seg_path = out_dir / f"{wav_path.stem}_seg{i:02d}.wav"
        sf.write(str(seg_path), clip, sr)
        out.append({"src": str(wav_path), "start": a, "end": b, "out": str(seg_path)})
    return out

# ------------------------ Artist Processing ------------------------
def process_single_artist(args) -> None:
    """Process a single artist (images, audio, segmentation)"""
    row, out_root, images_per_artist, clips_per_artist, segment, seg_min, seg_max = args
    
    name = row["name"]
    slug = slugify(name)
    base = out_root / slug
    img_dir = base / "images"
    aud_dir = base / "audio"
    ensure_dir(base)

    # Metadata
    meta = {
        "name": name, "qid": row.get("qid",""), "mbid": row.get("mbid",""),
        "country": row.get("country","")
    }
    (base / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"\n=== Processing {name} ===")
    
    # 2) Images
    print("  [2/4] Fetching photos...")
    df_img = fetch_images_for_artist(name, img_dir, max_images=images_per_artist, p18_url=row.get("image",""))
    if not df_img.empty:
        df_img.to_csv(base / "images.csv", index=False)
        print(f"  Saved {len(df_img)} images")
    else:
        print("  No available images")

    # 3) Audio
    print("  [3/4] Fetching and converting audio...")
    df_aud = fetch_audio_for_artist(name, aud_dir, max_clips=clips_per_artist)
    if not df_aud.empty:
        df_aud.to_csv(base / "audio.csv", index=False)
        print(f"  Saved {len(df_aud)} audio clips (original/full tracks)")
    else:
        print("  No available audio (consider expanding search or enabling Jamendo)")

    # 4) Segmentation
    if segment and not df_aud.empty:
        print("  [4/4] Segmenting voice clips...")
        seg_rows = []
        for f in df_aud["file"]:
            seg_rows += segment_audio_simple(pathlib.Path(f), aud_dir, seg_min, seg_max)
        if seg_rows:
            pd.DataFrame(seg_rows).to_csv(base / "audio_segments.csv", index=False)
            print(f"  Generated {len(seg_rows)} segments")
        else:
            print("  No valid segments generated (try adjusting thresholds or using smarter VAD models)")

# ------------------------ Main Pipeline ------------------------
def build_dataset(out_root: pathlib.Path, max_artists: int, images_per_artist: int, clips_per_artist: int,
                  segment: bool, seg_min: float, seg_max: float, max_workers: int = 4):
    """Build the complete dataset with parallel processing"""
    ensure_dir(out_root)
    print("[1/4] Fetching artist list (Wikidata)...")
    artists = fetch_artists_from_wikidata(max_artists=max_artists)
    if artists.empty:
        print("No artists retrieved.")
        return
    
    seed_csv = out_root / "artists_seed.csv"
    artists.to_csv(seed_csv, index=False)
    print(f"Artists found: {len(artists)} -> {seed_csv}")

    # Prepare processing tasks
    tasks = []
    for _, row in artists.iterrows():
        tasks.append((row, out_root, images_per_artist, clips_per_artist, segment, seg_min, seg_max))
    
    # Process artists in parallel
    print(f"\nProcessing {len(tasks)} artists with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_artist, task) for task in tasks]
        
        # Use tqdm for progress tracking
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing artists"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing artist: {e}")

# ------------------------ Entry Point ------------------------
def main():
    ap = argparse.ArgumentParser(description="Build popular singer (photos + voice) dataset")
    ap.add_argument("--out", default="singers", type=str, help="Output root directory")
    ap.add_argument("--max-artists", default=50, type=int, help="Maximum number of artists to fetch")
    ap.add_argument("--images-per-artist", default=10, type=int, help="Maximum images to save per artist")
    ap.add_argument("--clips-per-artist", default=6, type=int, help="Maximum audio clips to save per artist")
    ap.add_argument("--no-seg", action="store_true", help="Skip audio segmentation")
    ap.add_argument("--seg-min", default=6.0, type=float, help="Minimum segment length in seconds")
    ap.add_argument("--seg-max", default=12.0, type=float, help="Maximum segment length in seconds")
    ap.add_argument("--workers", default=4, type=int, help="Number of parallel workers")
    args = ap.parse_args()

    out_root = pathlib.Path(args.out)
    build_dataset(
        out_root=out_root,
        max_artists=args.max_artists,
        images_per_artist=args.images_per_artist,
        clips_per_artist=args.clips_per_artist,
        segment=not args.no_seg,
        seg_min=args.seg_min,
        seg_max=args.seg_max,
        max_workers=args.workers
    )
    print("\nCompleted. Please check each artist directory for images.csv / audio.csv / audio_segments.csv and meta.json.")

if __name__ == "__main__":
    main()