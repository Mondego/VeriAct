# responses.jsonl is the output file of running the baselines approaches
# --max-pairs is the max input/output test pairs to use in mutation

python -m spec_harness.eval_llm_response \
  --benchmark_path benchmarks/formalbench/fb.json \
  --llm_response_path path/to/responses.jsonl \
  --openjml openjml \
  --output spec_harness_results \
  --threads 8 \
  --max-pairs 5 \
  --verbose