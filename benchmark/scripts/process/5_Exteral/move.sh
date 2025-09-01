#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ====== 配置（可用参数覆盖）======
SRC_ROOT="${1:-/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/process/5_Exteral/pop_singers_2024_2025}"
DST_ROOT="${2:-/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/process/5_Exteral/pop_singers_yt_2024_2025}"
DRY_RUN="${DRY_RUN:-0}"   # 1=只打印，不复制
# 允许的音频后缀（大小写不敏感）
AUDIO_REGEX='.*\.\(wav\|mp3\|m4a\|webm\|flac\|opus\|ogg\)$'

# ====== 工具检测 ======
if ! command -v sha1sum >/dev/null 2>&1; then
  echo "[FATAL] 未找到 sha1sum，请在 Linux 上安装 coreutils。" >&2
  exit 1
fi

# ====== 函数 ======
log() { echo -e "$*"; }
warn() { echo -e "$*" >&2; }

# 生成不重名的目标文件路径（如已存在同名，则加 _1/_2/... 后缀）
unique_name() {
  local dir="$1" base="$2"
  local stem ext candidate i=1
  stem="${base%.*}"
  ext=".${base##*.}"
  candidate="$dir/$base"
  while [[ -e "$candidate" ]]; do
    candidate="$dir/${stem}_$i$ext"
    ((i++))
  done
  printf '%s' "$candidate"
}

# 为目标目录建立哈希索引（hash -> path）
build_hash_index() {
  local dir="$1"
  declare -gA HINDEX
  HINDEX=()
  if [[ ! -d "$dir" ]]; then return 0; fi
  local f
  # shellcheck disable=SC2044
  for f in $(find "$dir" -maxdepth 1 -type f -iregex "$AUDIO_REGEX" -print0 | xargs -0 -I{} echo "{}"); do
    local h
    if ! h="$(sha1sum "$f" | awk '{print $1}')" ; then
      warn "      [索引][WARN] 读取失败: $f"
      continue
    fi
    HINDEX["$h"]="$f"
  done
  return 0
}

copy_one_file() {
  local src="$1" dst_dir="$2"
  local h base dst
  h="$(sha1sum "$src" | awk '{print $1}')"
  if [[ -n "${HINDEX[$h]+x}" ]]; then
    log  "        [跳过][重复] $(basename "$src") （内容已存在：$(basename "${HINDEX[$h]}")）"
    return 0
  fi
  base="$(basename "$src")"
  dst="$(unique_name "$dst_dir" "$base")"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "        [DRY-RUN][复制] '$src' -> '$dst'"
  else
    cp -p "$src" "$dst"
    log "        [复制][OK ] $(basename "$src") -> $(basename "$dst")"
  fi
  # 更新索引，避免本轮内重复
  local newhash
  newhash="$h"
  HINDEX["$newhash"]="$dst"
}

sync_one_artist() {
  local artist_dir="$1"
  local artist="$(basename "$artist_dir")"
  local src_audio="$artist_dir/audio"
  local dst_artist_dir="$DST_ROOT/$artist"
  local dst_audio="$dst_artist_dir/audio"

  if [[ ! -d "$src_audio" ]]; then
    warn "  [$artist] 源无 audio/ ，跳过"
    return
  fi

  mkdir -p "$dst_audio"

  log "\n[$artist] 同步开始"
  # 目标已有文件索引
  log "  [准备] 扫描目标已有文件以做去重..."
  build_hash_index "$dst_audio"
  log "  [准备] 目标已有：${#HINDEX[@]} 个唯一文件（按 SHA1 判定）"

  # 列出源文件
  mapfile -d '' SRC_FILES < <(find "$src_audio" -maxdepth 1 -type f -iregex "$AUDIO_REGEX" -print0 | sort -z)
  if [[ "${#SRC_FILES[@]}" -eq 0 ]]; then
    warn "  [$artist] 源 audio/ 无音频文件，跳过"
    return
  fi

  local copied=0 skipped=0
  for srcf in "${SRC_FILES[@]}"; do
    # 去掉末尾的空字符
    srcf="${srcf%$'\0'}"
    if copy_one_file "$srcf" "$dst_audio"; then
      ((copied++))
    else
      ((skipped++))
    fi
  done
  log "[$artist] 同步完成：新复制 $copied，跳过(重复) $skipped"
}

# ====== 主流程 ======
main() {
  if [[ ! -d "$SRC_ROOT" ]]; then
    echo "[FATAL] 源根目录不存在：$SRC_ROOT" >&2; exit 1
  fi
  mkdir -p "$DST_ROOT"

  # 枚举源下的歌手目录（仅目录）
  mapfile -d '' ARTISTS < <(find "$SRC_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
  if [[ "${#ARTISTS[@]}" -eq 0 ]]; then
    echo "[FATAL] 源根目录下未找到任何歌手目录" >&2; exit 1
  fi

  log "[INFO] 源：$SRC_ROOT"
  log "[INFO] 目标：$DST_ROOT"
  if [[ "$DRY_RUN" == "1" ]]; then log "[INFO] DRY-RUN 模式：不会实际复制文件"; fi
  log "[INFO] 将处理歌手数：${#ARTISTS[@]}"

  local total_copied=0 total_skipped=0
  for adir in "${ARTISTS[@]}"; do
    adir="${adir%$'\0'}"
    sync_one_artist "$adir"
  done

  log "\n=== 全部完成 ==="
  log "源：$SRC_ROOT"
  log "目标：$DST_ROOT"
  log "（每位歌手的明细见上方日志）"
}

main "$@"