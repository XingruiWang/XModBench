#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time, argparse, pathlib, re, urllib.parse, requests
from dataclasses import dataclass
from typing import Optional, List, Dict
from tqdm import tqdm
import pandas as pd

UA = {"User-Agent": "AlbumCoverFetcher/1.0 (mailto:example@example.com)"}
MBZ_SEARCH = "https://musicbrainz.org/ws/2/release/"
CAA_FRONT = "https://coverartarchive.org/release/{mbid}/front-{size}.jpg"  # size: 250/500/1200

GENRES = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

@dataclass
class Release:
    mbid: str
    title: str
    artist: str
    date: Optional[str]
    score: int

def slugify(s: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|]+", "", s)
    s = re.sub(r"\s+", "_", s.strip())
    return s[:140]

def search_releases_by_tag(tag: str, limit: int=100) -> List[Release]:
    """
    在 MusicBrainz 搜索：带有 tag 的正式专辑（primary-type:album），优先有日期和高得分。
    说明：MBZ 不保证“genre”字段全；tag/disambiguation/annotation里大量写法，这里以 tag: 做召回。
    """
    # q 里用 tag:xxx AND primarytype:album
    params = {
        "query": f'tag:"{tag}" AND primarytype:album',
        "fmt": "json",
        "limit": str(limit),
        "inc": "artist-credits",
    }
    r = requests.get(MBZ_SEARCH, params=params, headers=UA, timeout=30)
    r.raise_for_status()
    js = r.json()
    out: List[Release] = []
    for item in js.get("releases", []):
        mbid = item.get("id")
        title = item.get("title","")
        # 取主艺人名
        acs = item.get("artist-credit", [])
        artist = "".join(a.get("name","") if isinstance(a,dict) else str(a) for a in acs)
        date = item.get("date")
        score = int(item.get("score", 0))
        if mbid and title:
            out.append(Release(mbid, title, artist, date, score))
    # 简单排序：先 score，再发布日期（新到旧），再标题
    out.sort(key=lambda x: (x.score, x.date or "0000"), reverse=True)
    return out

def try_download(url: str, path: pathlib.Path) -> bool:
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with requests.get(url, headers=UA, stream=True, timeout=30) as r:
            if r.status_code == 404:
                return False
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk: f.write(chunk)
        tmp.rename(path)
        return True
    except Exception:
        tmp.exists() and tmp.unlink()
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="covers_mbz", help="输出根目录")
    ap.add_argument("--per_genre", type=int, default=100, help="每个流派保存前 N 张")
    ap.add_argument("--thumb", type=int, default=1200, help="封面边长(250/500/1200)")
    ap.add_argument("--sleep", type=float, default=1.1, help="MusicBrainz 频率限制(>=1s)")
    args = ap.parse_args()

    root = pathlib.Path(args.out); root.mkdir(parents=True, exist_ok=True)
    for g in GENRES:
        print(f"\n=== {g} ===")
        rels = search_releases_by_tag(g, limit=max(100, args.per_genre*3))
        gdir = root / g; gdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx, rel in enumerate(tqdm(rels[:args.per_genre], desc=f"{g}")):
            # CAA front 封面
            url = CAA_FRONT.format(mbid=rel.mbid, size=args.thumb)
            fname = f"{idx+1:02d}_{slugify(rel.artist)}-{slugify(rel.title)}"
            if rel.date: fname += f"_{rel.date[:4]}"
            path = gdir / (fname + ".jpg")
            ok = try_download(url, path)
            if not ok:
                # 退一步，用 500 像素
                url2 = CAA_FRONT.format(mbid=rel.mbid, size=500)
                ok = try_download(url2, path)
            rows.append({
                "rank": idx+1,
                "genre": g,
                "mbid": rel.mbid,
                "title": rel.title,
                "artist": rel.artist,
                "date": rel.date,
                "saved": ok,
                "url_tried": url,
            })
            # 遵守 MBZ 限频：每次循环 sleep
            time.sleep(args.sleep)
        pd.DataFrame(rows).to_csv(gdir / f"metadata_{g}.csv", index=False)
        print(f"{g}: 保存成功 {sum(r['saved'] for r in rows)} 张，目录：{gdir}")

if __name__ == "__main__":
    main()
