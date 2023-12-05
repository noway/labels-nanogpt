#!/bin/bash

SOUND_FILE="/System/Library/Sounds/Funk.aiff"
getCurrentOutput() {
    ./track_instance_types.sh | jq '.data.gpu_1x_h100_pcie.regions_with_capacity_available | length'
}
previous_output=$(getCurrentOutput)
while true; do
    current_output=$(getCurrentOutput)
    echo "Current output: $current_output"
    if [ "$current_output" != "$previous_output" ]; then
        afplay "$SOUND_FILE"
        previous_output=$current_output
    fi
    sleep 1
done
