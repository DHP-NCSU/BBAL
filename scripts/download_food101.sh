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
TEMP_DIR="./downloads/food101"
mkdir -p "$TEMP_DIR"

echo "Downloading Food-101 dataset..."
cd "$TEMP_DIR"
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz

echo "Extracting dataset..."
tar -xzf food-101.tar.gz
rm food-101.tar.gz

echo "Setting up directories..."
cd ../../
mkdir -p data/food101/train
mkdir -p data/food101/test

echo "Moving dataset..."
mv "$TEMP_DIR"/food-101/images/* data/food101/train/
mv "$TEMP_DIR"/food-101/meta data/food101/
rm -rf "$TEMP_DIR"

echo "Splitting dataset..."
cd data/food101

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