#!/usr/bin/env bash
set -euo pipefail

NUM_PERMUTATIONS="${1:-10000}"
SEED="${2:-42}"

echo "=== Permutation Test: All Models ==="
echo "Permutations: $NUM_PERMUTATIONS | Seed: $SEED"
echo ""

for results_file in wvs_values_results/*/profile_values_results/total_1000.jsonl; do
    model_dir=$(dirname "$(dirname "$results_file")")
    model_name=$(basename "$model_dir")
    output_path="$model_dir/permutation_test_results.json"

    echo "--- $model_name ---"
    python scripts/permutation_test_homogenization.py \
        --ba-user-results "$results_file" \
        --output-path "$output_path" \
        --num-permutations "$NUM_PERMUTATIONS" \
        --seed "$SEED"
    echo ""
done

echo "=== Summary ==="
for results_file in wvs_values_results/*/permutation_test_results.json; do
    model_name=$(basename "$(dirname "$results_file")")
    python -c "
import json, sys
with open('$results_file') as f:
    d = json.load(f)
s = d['summary']
print(f'  {\"$model_name\":40s} all_sig={s[\"all_significant\"]}  effect={s[\"mean_demographic_effect\"]:.4f}  p=[{s[\"min_p_value\"]:.4f}, {s[\"max_p_value\"]:.4f}]')
"
done
