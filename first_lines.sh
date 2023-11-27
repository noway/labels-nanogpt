#!/bin/bash

for file in *.txt; do
    if [ -f "$file" ]; then
        first_line=$(head -n 1 "$file")
        echo "$first_line"
    fi
done
