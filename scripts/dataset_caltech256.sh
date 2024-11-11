#!/bin/bash

# Detect OS and use different shuf command
if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v gshuf >/dev/null 2>&1; then
        shuffle_cmd="gshuf"
    else
        echo "Please install gshuf using 'brew install coreutils'"
        exit 1
    fi
else
    shuffle_cmd="shuf"
fi

# Download the dataset
echo "Downloading dataset..."
output=$(python utils/download.py)
dataset_path=$(echo "$output" | tail -n 1 | sed 's/Path to dataset files: //')

# Create directories
echo "Setting up directories..."
mkdir -p data/caltech256/train
mkdir -p data/caltech256/test

# Move the downloaded dataset - fixed path handling
echo "Moving downloaded dataset..."
mv "$dataset_path"/256_ObjectCategories/* data/caltech256/train/
rm -rf "$dataset_path"

# Split the dataset
echo "Splitting dataset..."
cd data/caltech256 || exit 1

for dir in train/*; do
    if [ ! -d "$dir" ]; then
        continue
    fi
    
    category=$(basename "$dir")
    # echo "Processing category: $category"
    mkdir -p "test/$category"
    
    # Calculate number of files to move
    total_files=$(ls -1 "$dir" | wc -l)
    files_to_move=$(( total_files * 20 / 100 ))
    
    # Move files in one operation
    ls -1 "$dir" | $shuffle_cmd -n "$files_to_move" | \
    while read -r file; do
        mv "$dir/$file" "test/$category/"
    done
done

echo "Dataset split complete!"