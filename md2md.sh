#!/bin/bash

SOURCE_DIR="dataset"
OUTPUT_DIR="dataset_md2md"
[ ! -d "$OUTPUT_DIR" ] && mkdir -p "$OUTPUT_DIR"
for file in "$SOURCE_DIR"/*.txt; do
    filename=$(basename "$file")
    pandoc -f markdown -t markdown -o "$OUTPUT_DIR/$filename" "$file"
done
