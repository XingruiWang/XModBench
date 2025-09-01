#!/bin/bash
set -euo pipefail

FFMPEG="/usr/bin/ffmpeg"
FFPROBE="/usr/bin/ffprobe"
# 修改为你的根目录路径
ROOT_DIR="/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/process/2_spatial_audio/output/questions_video_choice"

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

echo "[start] resuming batch re-encoding in nested 'test' folders: $ROOT_DIR"
echo "========================================================"

# 全局统计变量
total_processed=0
total_pending=0
total_errors=0
total_skipped=0
total_folders=0

# 分析单个文件夹状态的函数
analyze_folder_state() {
    local folder="$1"
    local folder_name=$(basename "$folder")
    
    echo "[analyze] checking folder: $folder_name"
    
    local completed_files=0
    local pending_files=0
    local ori_files=0
    
    # 统计所有mp4文件的状态
    while IFS= read -r -d '' mp4_file; do
        local base="${mp4_file%.mp4}"
        local ori_file="${base}_ori.mp4"
        
        # 如果有对应的_ori文件，就算已完成（跳过）
        if [[ -f "$ori_file" ]]; then
            ori_files=$((ori_files + 1))
            completed_files=$((completed_files + 1))
        else
            # 没有_ori文件，需要处理
            pending_files=$((pending_files + 1))
        fi
    done < <(find "$folder" -maxdepth 1 -name "*.mp4" -not -name "*_ori.mp4" -type f -print0 2>/dev/null || true)
    
    echo "  - Files with _ori.mp4: $ori_files"
    echo "  - Completed conversions: $completed_files" 
    echo "  - Pending conversions: $pending_files"
    
    echo "$pending_files"
}

# 处理单个文件的函数
process_video() {
    local f="$1"
    local base="${f%.mp4}"
    local ori="${base}_ori.mp4"
    local out="${base}.mp4"
    local rel_path=$(realpath --relative-to="$ROOT_DIR" "$f")
    
    echo
    echo "[processing] $rel_path"
    echo "----------------------------------------"
    
    # 检查是否已经是_ori.mp4文件，如果是则跳过
    if [[ "$f" == *"_ori.mp4" ]]; then
        echo "[skip] this is an _ori file: $(basename "$f")"
        return 2  # 特殊返回码表示跳过
    fi
    
    # 处理文件备份逻辑
    if [[ -f "$ori" ]]; then
        echo "[found] using existing _ori file: $(basename "$ori")"
        # 检查原文件是否还存在，如果存在且不同于ori文件，则删除
        if [[ -f "$f" && "$f" != "$ori" ]]; then
            echo "[cleanup] removing redundant original file: $(basename "$f")"
            rm "$f"
        fi
    else
        if [[ -f "$f" ]]; then
            echo "[backup] $(basename "$f") -> $(basename "$ori")"
            mv "$f" "$ori"
        else
            echo "[error] source file not found: $f"
            return 1
        fi
    fi
    
    # 检查输出文件是否已存在且有效
    if [[ -f "$out" ]]; then
        if existing_codec=$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=codec_name \
                           -of default=nw=1:nk=1 "$out" 2>/dev/null); then
            if [[ "$existing_codec" == "h264" ]]; then
                echo "[already done] output already exists with H.264 codec: $(basename "$f")"
                return 0
            else
                echo "[redo] output exists but codec is $existing_codec, re-encoding..."
            fi
        else
            echo "[redo] output exists but seems corrupted, re-encoding..."
        fi
    fi
    
    # 获取原视频信息
    echo "[analyze] getting video info from _ori file..."
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
    resolution=$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=width,height \
                 -of csv=s=x:p=0 "$ori" 2>/dev/null || echo "unknown")
    
    echo "[info] fps: $orig_fps, codec: $orig_codec, resolution: $resolution, duration: ${duration}s"
    
    # 重新编码视频
    echo "[ffmpeg] re-encoding to H.264..."
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
            
            echo "[result] codec: $final_codec, fps: $final_fps, duration: ${final_duration}s"
            
            if [[ "$final_codec" == "h264" ]]; then
                echo "[success] re-encoding completed: $(basename "$f")"
                return 0
            else
                echo "[warning] codec not H.264: $final_codec"
                return 1
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

# 处理单个文件夹的函数
process_folder() {
    local folder="$1"
    local folder_name=$(basename "$folder")
    local folder_rel_path=$(realpath --relative-to="$ROOT_DIR" "$folder")
    
    echo
    echo "=========================================="
    echo "[folder] processing: $folder_rel_path"
    echo "=========================================="
    
    # 检查文件夹中是否有mp4文件
    local mp4_count=$(find "$folder" -maxdepth 1 -name "*.mp4" -type f | wc -l)
    
    if [[ $mp4_count -eq 0 ]]; then
        echo "[info] no MP4 files found in: $folder_name"
        return 0
    fi
    
    # 分析当前状态
    local pending_count
    pending_count=$(analyze_folder_state "$folder")
    # if analyze_folder_state "$folder"; then
    #     pending_count=$?
    # else
    #     pending_count=0
    # fi
    
    if [[ $pending_count -eq 0 ]]; then
        echo "[skip] all files in $folder_name are already processed"
        return 0
    fi
    
    echo "[info] processing $pending_count pending files in: $folder_name"
    
    local folder_processed=0
    local folder_errors=0
    local folder_skipped=0
    local file_count=0
    
    # Phase 1: 处理有_ori文件的情况
    echo
    echo "[phase 1] processing files with existing _ori files..."
    while IFS= read -r -d '' ori_file; do
        local base="${ori_file%_ori.mp4}"
        local normal_file="${base}.mp4"
        
        file_count=$((file_count + 1))
        echo "  File $file_count: $(basename "$normal_file")"
        
        local result
        if process_video "$normal_file"; then
            folder_processed=$((folder_processed + 1))
            total_processed=$((total_processed + 1))
        elif [[ $? -eq 2 ]]; then
            folder_skipped=$((folder_skipped + 1))
            total_skipped=$((total_skipped + 1))
        else
            folder_errors=$((folder_errors + 1))
            total_errors=$((total_errors + 1))
        fi
        
    done < <(find "$folder" -maxdepth 1 -name "*_ori.mp4" -type f -print0 2>/dev/null | sort -z || true)
    
    # Phase 2: 处理还没有_ori文件的原始文件
    echo
    echo "[phase 2] processing files without _ori files..."
    while IFS= read -r -d '' mp4_file; do
        local base="${mp4_file%.mp4}"
        local ori_file="${base}_ori.mp4"
        
        if [[ ! -f "$ori_file" ]]; then
            file_count=$((file_count + 1))
            echo "  File $file_count: $(basename "$mp4_file")"
            
            if process_video "$mp4_file"; then
                folder_processed=$((folder_processed + 1))
                total_processed=$((total_processed + 1))
            elif [[ $? -eq 2 ]]; then
                folder_skipped=$((folder_skipped + 1))
                total_skipped=$((total_skipped + 1))
            else
                folder_errors=$((folder_errors + 1))
                total_errors=$((total_errors + 1))
            fi
        fi
        
    done < <(find "$folder" -maxdepth 1 -name "*.mp4" -not -name "*_ori.mp4" -type f -print0 2>/dev/null | sort -z || true)
    
    echo
    echo "[folder summary] $folder_name: processed=$folder_processed, errors=$folder_errors, skipped=$folder_skipped"
}

# 主处理循环 - 递归查找所有包含MP4文件且含有"test"的文件夹
echo "[scan] scanning for folders with MP4 files (containing 'test' in path)..."

# 查找所有包含mp4文件且路径中含有"test"的文件夹
while IFS= read -r -d '' folder; do
    # 检查文件夹路径是否包含"test"
    if [[ "$folder" == *"test"* ]]; then
        total_folders=$((total_folders + 1))
        process_folder "$folder"
    else
        echo "[skip] folder does not contain 'test': $(realpath --relative-to="$ROOT_DIR" "$folder")"
    fi
done < <(find "$ROOT_DIR" -type d -exec sh -c 'ls "$1"/*.mp4 >/dev/null 2>&1' _ {} \; -print0 | sort -z)

# 输出最终统计结果
echo
echo "========================================================"
echo "[FINAL SUMMARY] nested folder processing completed"
echo "  Folders processed: $total_folders"
echo "  Successfully processed: $total_processed"
echo "  Skipped (already done): $total_skipped" 
echo "  Errors: $total_errors"
echo "========================================================"

# 最终验证 - 检查所有含有"test"的文件夹的状态
echo
echo "[verification] final state check across all 'test' folders..."
total_h264_files=0
total_output_files=0

while IFS= read -r -d '' folder; do
    # 只验证含有"test"的文件夹
    if [[ "$folder" == *"test"* ]]; then
        folder_h264=$(find "$folder" -maxdepth 1 -name "*.mp4" -not -name "*_ori.mp4" -type f -exec sh -c 'codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nw=1:nk=1 "$1" 2>/dev/null); [[ "$codec" == "h264" ]]' _ {} \; -print | wc -l)
        folder_total=$(find "$folder" -maxdepth 1 -name "*.mp4" -not -name "*_ori.mp4" -type f | wc -l)
        
        if [[ $folder_total -gt 0 ]]; then
            folder_rel=$(realpath --relative-to="$ROOT_DIR" "$folder")
            echo "  $folder_rel: $folder_h264/$folder_total files are H.264"
            total_h264_files=$((total_h264_files + folder_h264))
            total_output_files=$((total_output_files + folder_total))
        fi
    fi
    
done < <(find "$ROOT_DIR" -type d -exec sh -c 'ls "$1"/*.mp4 >/dev/null 2>&1' _ {} \; -print0 | sort -z)

echo
echo "[final result] $total_h264_files/$total_output_files total files in 'test' folders are now H.264 encoded"

if [[ $total_errors -gt 0 ]]; then
    echo "[warning] $total_errors files had errors"
    exit 1
else
    echo "[success] all processing completed successfully"
    exit 0
fi