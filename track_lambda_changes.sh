#!/bin/bash

source .env
SOUND_FILE="/System/Library/Sounds/Funk.aiff"
getCurrentOutput() {
    ./track_instance_types.sh | jq '.data[] | select(.regions_with_capacity_available != []).instance_type.name'
}
previous_output=$(getCurrentOutput)
while true; do
    current_output=$(getCurrentOutput)
    echo "Current output: $current_output"
    if [ "$current_output" != "$previous_output" ]; then
        afplay "$SOUND_FILE" &
        osascript -e 'display notification "Any capacity has changed" with title "Notification" subtitle "Please attend"'
        current_output_processed="$(echo $current_output | tr -d '\"')"
        curl -X POST -H "Content-Type: application/json" -d "{ \"username\": \"lambda-tracker\", \"content\": \"<@$DISCORD_USER_ID> capacity has changed: $current_output_processed\" }" $DISCORD_WEBHOOK_URL
    fi
    previous_output=$current_output
    sleep 1
done
