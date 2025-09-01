#!/bin/bash
set -euo pipefail

FFMPEG="/usr/bin/ffmpeg"
FFPROBE="/usr/bin/ffprobe"
# 修改为你的ExtremCountAV文件夹路径
ROOT_DIR="/home/xwang378/scratch/2025/AudioBench/benchmark/Data/ExtremCountAV"

# 检查工具是否存在
for tool in "$FFMPEG" "$FFPROBE"; do
    if [[ ! -x "$tool" ]]; then
        echo "[error] $tool not found or not executable"
        exit 1
    fi
done

# 检查根目录是否存在
if [[ ! -d "$ROOT_DIR" ]]; then
    echo "[error] root directory not found: $ROOT_DIR"
    exit 1
fi

echo "[start] batch re-encoding videos in: $ROOT_DIR"
echo "========================================================"

# 统计变量
total_files=0
processed_files=0
skipped_files=0
error_files=0

# 处理单个文件的函数
process_video() {
    local f="$1"
    local filename=$(basename "$f")
    
    # 正确处理文件名
    if [[ "$f" == *"_ori.mp4" ]]; then
        # 如果输入文件已经是_ori文件
        local base="${f%_ori.mp4}"
        local ori="$f"
        local out="${base}.mp4"
    else
        # 如果输入文件是普通mp4文件
        local base="${f%.mp4}"
        local ori="${base}_ori.mp4"
        local out="${base}.mp4"
    fi
    
    echo
    echo "[processing] $filename"
    echo "----------------------------------------"
    
    # 如果两个文件都存在，则跳过
    if [[ -f "$ori" && -f "$out" ]]; then
        echo "[skip] both files already exist: $(basename "$ori") and $(basename "$out")"
        return 1
    fi
    
    # 如果输入文件是_ori.mp4，需要调整变量
    if [[ "$f" == *"_ori.mp4" ]]; then
        # 输入就是_ori文件，不需要备份
        echo "[info] processing _ori file: $filename"
    fi
    
    # 处理文件备份逻辑
    if [[ "$f" == *"_ori.mp4" ]]; then
        # 输入文件本身就是_ori文件
        if [[ ! -f "$f" ]]; then
            echo "[error] _ori file not found: $f"
            return 1
        fi
        echo "[info] using _ori file as source: $(basename "$f")"
    else
        # 输入文件是普通的.mp4文件
        if [[ -f "$ori" ]]; then
            echo "[found] using existing _ori file: $(basename "$ori")"
            # 如果原始文件不存在，说明需要重新编码
            if [[ ! -f "$out" ]]; then
                echo "[info] output file missing, will re-encode"
            fi
        else
            if [[ -f "$f" ]]; then
                echo "[backup] $filename -> $(basename "$ori")"
                mv "$f" "$ori"
            else
                echo "[error] source file not found: $f"
                return 1
            fi
        fi
    fi
    
    # 获取原视频信息
    echo "[analyze] getting video info..."
    if ! orig_fps=$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=r_frame_rate \
                   -of default=nw=1:nk=1 "$ori" 2>/dev/null); then
        echo "[error] failed to get frame rate from: $(basename "$ori")"
        return 1
    fi
    
    if ! duration=$("$FFPROBE" -v error -show_entries format=duration \
                   -of default=nw=1:nk=1 "$ori" 2>/dev/null); then
        echo "[error] failed to get duration from: $(basename "$ori")"
        return 1
    fi
    
    orig_codec=$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=codec_name \
                 -of default=nw=1:nk=1 "$ori" 2>/dev/null || echo "unknown")
    
    # 获取分辨率信息
    resolution=$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=width,height \
                 -of csv=s=x:p=0 "$ori" 2>/dev/null || echo "unknown")
    
    echo "[info] fps: $orig_fps, codec: $orig_codec, resolution: $resolution, duration: ${duration}s"

    
    # 重新编码视频 - 保持原有帧率和时长
    echo "[ffmpeg] re-encoding to H.264 (preserving all original specs)..."
    if "$FFMPEG" -y -i "$ori" \
        -c:v libx264 -preset medium -crf 23 \
        -pix_fmt yuv420p -profile:v high -level 4.0 -tag:v avc1 \
        -an -movflags +faststart \
        "$out" 2>/dev/null; then
        
        # 验证转换结果
        if final_codec=$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=codec_name \
                        -of default=nw=1:nk=1 "$out" 2>/dev/null); then
            final_fps=$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=r_frame_rate \
                       -of default=nw=1:nk=1 "$out" 2>/dev/null || echo "unknown")
            final_duration=$("$FFPROBE" -v error -show_entries format=duration \
                            -of default=nw=1:nk=1 "$out" 2>/dev/null || echo "unknown")
            final_resolution=$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=width,height \
                              -of csv=s=x:p=0 "$out" 2>/dev/null || echo "unknown")
            
            echo "[result] codec: $final_codec, fps: $final_fps"
            echo "         resolution: $final_resolution, duration: ${final_duration}s"
            
            if [[ "$final_codec" == "h264" ]]; then
                echo "[success] re-encoding completed: $filename"
                return 0
            else
                echo "[warning] codec not H.264: $final_codec"
                return 0
            fi
        else
            echo "[error] failed to verify output file"
            return 1
        fi
    else
        echo "[error] ffmpeg re-encoding failed"
        return 1
    fi
}

# 主处理循环 - 处理单层目录中的所有mp4文件
echo "[scan] looking for MP4 files in: $ROOT_DIR"

# 收集所有需要处理的文件（只收集_ori.mp4文件和没有对应_ori版本的.mp4文件）
declare -a files_to_process=()
while IFS= read -r -d '' file; do
    base="${file%.mp4}"
    if [[ "$file" == *"_ori.mp4" ]]; then
        # 如果是_ori文件，检查是否缺少对应的.mp4文件
        base="${file%_ori.mp4}"
        regular_file="${base}.mp4"
        if [[ ! -f "$regular_file" ]]; then
            files_to_process+=("$file")
        fi
    else
        # 如果是普通.mp4文件，检查是否有对应的_ori文件
        ori_file="${base}_ori.mp4"
        if [[ ! -f "$ori_file" ]]; then
            files_to_process+=("$file")
        fi
    fi
done < <(find "$ROOT_DIR" -maxdepth 1 -name "*.mp4" -type f -print0 | sort -z)

total_mp4_files=${#files_to_process[@]}
echo "[info] found $total_mp4_files files that need processing"

if [[ $total_mp4_files -eq 0 ]]; then
    echo "[info] no files need processing - all files already have both versions"
    exit 0
fi

echo
echo "Starting processing..."

# 处理收集到的文件
for file in "${files_to_process[@]}"; do
    total_files=$((total_files + 1))
    
    echo "Progress: $total_files/$total_mp4_files"
    
    if process_video "$file"; then
        processed_files=$((processed_files + 1))
    else
        if [[ "$file" == *"_ori.mp4" ]]; then
            skipped_files=$((skipped_files + 1))
        else
            error_files=$((error_files + 1))
        fi
    fi
done

# 输出最终统计结果
echo
echo "========================================================"
echo "[FINAL SUMMARY] batch re-encoding completed"
echo "  Total files found: $total_files"
echo "  Successfully processed: $processed_files"
echo "  Skipped (_ori files): $skipped_files"
echo "  Errors: $error_files"
echo "========================================================"

# 列出处理后的文件
echo
echo "[FILES] Processed video files:"
find "$ROOT_DIR" -maxdepth 1 -name "*.mp4" -not -name "*_ori.mp4" -type f | sort | while read -r file; do
    filename=$(basename "$file")
    codec=$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=codec_name \
            -of default=nw=1:nk=1 "$file" 2>/dev/null || echo "unknown")
    fps=$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=r_frame_rate \
          -of default=nw=1:nk=1 "$file" 2>/dev/null || echo "unknown")
    echo "  ✓ $filename (codec: $codec, fps: $fps)"
done

if [[ $error_files -gt 0 ]]; then
    echo
    echo "[warning] $error_files files had errors"
    exit 1
else
    echo
    echo "[done] all files re-encoded successfully to H.264"
    exit 0
fi