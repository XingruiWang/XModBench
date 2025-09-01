#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="$1"   # 源根目录（老数据集）
DST_ROOT="$2"   # 目标根目录（新数据集）
DRY_RUN="${DRY_RUN:-0}"  # 1=演练模式只打印

for SRC_ARTIST_DIR in "$SRC_ROOT"/*; do
  [[ -d "$SRC_ARTIST_DIR/audio" ]] || continue
  ARTIST="$(basename "$SRC_ARTIST_DIR")"
  DST_ARTIST_DIR="$DST_ROOT/$ARTIST/audio"
  mkdir -p "$DST_ARTIST_DIR"

  echo "[$ARTIST] 移动中..."
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  rsync -a --ignore-existing --remove-source-files \"$SRC_ARTIST_DIR/audio/\" \"$DST_ARTIST_DIR/\""
  else
    rsync -a --ignore-existing --remove-source-files "$SRC_ARTIST_DIR/audio/" "$DST_ARTIST_DIR/"
    # 清理源端空目录（可选）
    find "$SRC_ARTIST_DIR/audio" -type d -empty -delete || true
  fi
done

echo "=== 完成 ==="
