#!/usr/bin/env bash
set -euo pipefail


response_list_file="${1:?Usage: $0 spec_har_sgb_run_jsonl_test.txt}"

basepath="/home/mdrh/experiments/all_approaches_result_jsonl"
benchmark_path="/home/mdrh/code/VeriAct/benchmarks/specgenbench/sgb.json"
openjml="openjml"
output_root="/home/mdrh/experiments/spec_harness_sgb_results_updated"
threads=12
max_pairs=5

mkdir -p "$output_root"

while IFS= read -r llm_response_path || [[ -n "$llm_response_path" ]]; do
  [[ -z "$llm_response_path" ]] && continue
  [[ "$llm_response_path" =~ ^# ]] && continue

  output_name="$(basename "$llm_response_path")"
  output_name="${output_name%.jsonl}"

  python -m spec_harness.eval_llm_response \
    --benchmark_path "$benchmark_path" \
    --llm_response_path "$basepath/$llm_response_path" \
    --openjml "$openjml" \
    --output "$output_root/$output_name" \
    --threads "$threads" \
    --max-pairs "$max_pairs" \
    --verbose
  
  sleep 5
  pkill -f cvc4-1.6 || true
  sleep 5

done < "$response_list_file"