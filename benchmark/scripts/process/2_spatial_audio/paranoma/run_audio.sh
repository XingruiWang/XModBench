#!/bin/bash

# batch_process_spatial_audio.sh
# Script to batch process STARSS23 spatial audio data

# Configuration
SOFA_FILE="subject_008.sofa"  # Update this path to your SOFA file
INPUT_ROOT="/home/xwang378/scratch/2025/AudioBench/benchmark/Data/STARSS23_processed"  # Update this to your root directory
OUTPUT_ROOT="/home/xwang378/scratch/2025/AudioBench/benchmark/Data/STARSS23_processed_augmented"  # Update this to your desired output directory

# Create output directory
mkdir -p "$OUTPUT_ROOT"

# Find and process all metadata files
echo "Starting spatial audio processing..."
echo "Input directory: $INPUT_ROOT"
echo "Output directory: $OUTPUT_ROOT"
echo "SOFA file: $SOFA_FILE"
echo ""

# Check if SOFA file exists
if [ ! -f "$SOFA_FILE" ]; then
    echo "Error: SOFA file not found: $SOFA_FILE"
    echo "Please update the SOFA_FILE path in this script"
    exit 1
fi

# Run the Python processor
python3 augment_spatial_audio.py \
    --input_dir "$INPUT_ROOT" \
    --sofa "$SOFA_FILE" \
    --output_dir "$OUTPUT_ROOT"

echo ""
echo "Processing completed!"
echo "Check $OUTPUT_ROOT/high_quality.txt for the list of high-quality events"