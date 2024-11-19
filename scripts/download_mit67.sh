#!/bin/bash

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

# Create a temporary directory for downloads
TEMP_DIR="./downloads/mit67"
mkdir -p "$TEMP_DIR"

echo "Downloading MIT Indoor-67 dataset..."
cd "$TEMP_DIR"

# Download dataset
curl -L -o indoor.zip https://www.kaggle.com/api/v1/datasets/download/itsahmad/indoor-scenes-cvpr-2019

echo "Extracting dataset..."
unzip -q indoor.zip

echo "Setting up directories..."
cd ../../
mkdir -p data/mit67/train
mkdir -p data/mit67/test

echo "Moving dataset..."
# The images should be in indoorCVPR_09/Images/
if [ -d "$TEMP_DIR/indoorCVPR_09/Images" ]; then
    echo "Found image directory, moving files..."
    mv "$TEMP_DIR/indoorCVPR_09/Images"/* data/mit67/train/
else
    echo "Error: Could not find image directory in expected location"
    echo "Contents of download directory:"
    ls -R "$TEMP_DIR"
    exit 1
fi

# Clean up
rm -rf "$TEMP_DIR"

echo "Splitting dataset..."
cd data/mit67

for dir in train/*; do
    if [ ! -d "$dir" ]; then
        continue
    fi
    
    category=$(basename "$dir")
    mkdir -p "test/$category"
    
    total_files=$(ls -1 "$dir" | wc -l)
    files_to_move=$(( total_files * 20 / 100 ))
    
    ls -1 "$dir" | $shuffle_cmd -n "$files_to_move" | \
    while read -r file; do
        mv "$dir/$file" "test/$category/"
    done
done

echo "Dataset split complete!"