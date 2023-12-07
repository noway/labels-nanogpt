#!/bin/bash

SOUND_FILE="/System/Library/Sounds/Funk.aiff"
getCurrentOutput() {
    ./track_instance_types.sh | jq '.data.gpu_1x_h100_pcie.regions_with_capacity_available'
}
previous_output=$(getCurrentOutput)
while true; do
    current_output=$(getCurrentOutput)
    echo "Current output: $current_output"
    if [ "$current_output" != "[]" ]; then
        afplay "$SOUND_FILE" &
        osascript -e 'display notification "Capacity has changed" with title "Notification" subtitle "Please attend"'
    fi
    previous_output=$current_output
    sleep 1
done
