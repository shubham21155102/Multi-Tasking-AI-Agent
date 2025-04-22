#!/bin/bash

# Usage: ./trim-audio.sh input_file start_time end_time output_file
# Example: ./trim-audio.sh src/services/audio/meet.m4a 00:01:00 00:02:00 src/services/audio/output.m4a

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 input_file start_time end_time output_file"
    echo "Times should be in HH:MM:SS format"
    exit 1
fi

INPUT_FILE=$1
START_TIME=$2
END_TIME=$3
OUTPUT_FILE=$4

# Create output directory if it doesn't exist
mkdir -p $(dirname "$OUTPUT_FILE")

# Run ffmpeg
ffmpeg -ss $START_TIME -to $END_TIME -i "$INPUT_FILE" -acodec copy "$OUTPUT_FILE" -y

echo "Audio trimmed successfully from $START_TIME to $END_TIME"
echo "Output saved to $OUTPUT_FILE"