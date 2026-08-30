#!/usr/bin/env bash
# Runs wvs_values_prediction with automatic resume on failure.
# On each crash, updates starting_row to current output line count and retries.
# If no progress was made (stuck user), skips ahead by 1.
# Usage: api_key=... bash scripts/run_with_resume.sh <config.yaml> <output.jsonl>
set -euo pipefail

CONFIG="$1"
OUTPUT="$2"
MAX_RETRIES=80
SLEEP_ON_FAIL=30  # seconds; lets DeepInfra rate limit clear before retry

for attempt in $(seq 1 $MAX_RETRIES); do
    COUNT=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
    echo "[resume] attempt=$attempt starting_row=$COUNT"
    sed -i "" "s/^starting_row:.*$/starting_row: $COUNT/" "$CONFIG"

    if api_key="$api_key" python -m llm_behavior_adaptation.value_measurement.wvs_values_prediction \
        --config "$CONFIG" 2>&1; then
        echo "[resume] completed successfully"
        exit 0
    fi

    NEW_COUNT=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
    if [ "$NEW_COUNT" -eq "$COUNT" ]; then
        echo "[resume] no progress at row $COUNT, skipping to $((COUNT + 1))"
        sed -i "" "s/^starting_row:.*$/starting_row: $((COUNT + 1))/" "$CONFIG"
    fi
    echo "[resume] sleeping ${SLEEP_ON_FAIL}s before retry..."
    sleep $SLEEP_ON_FAIL
done

echo "[resume] exhausted $MAX_RETRIES retries" >&2
exit 1
