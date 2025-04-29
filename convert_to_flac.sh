
#!/usr/bin/env bash

# convert_to_flac.sh
# Description: Batch-convert WAV files in Animal/ to FLAC in Animal_clean/, preserving class subfolders.

# Ensure prerequisites are installed:
# sudo apt update
# sudo apt install ffmpeg libsndfile1

# Run this script from your project root (where Animal/ resides).

# Create the clean output directory
#!/usr/bin/env bash
mkdir -p Animal_clean
for cls in Animal/*/; do
  class_name=$(basename "$cls")
  mkdir -p Animal_clean/"$class_name"
  for f in "$cls"/*.wav; do
    filename=$(basename "$f" .wav)
    ffmpeg -v error \
           -i "$f" \
           -acodec flac \
           -ar 22050 \
           -ac 1 \
           Animal_clean/"$class_name"/"$filename.flac"
  done
done
