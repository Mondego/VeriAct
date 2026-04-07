# Spec-Harness


Spec-Harness evaluates four metrics on verifier-accepted specifications:

| Metric | What it measures |
|--------|-----------------|
| PostCorrectness | Postcondition holds on valid test pairs |
| PostCompleteness | Postcondition catches failing test pairs |
| PreCorrectness | Precondition accepts valid inputs |
| PreCompleteness | Precondition rejects invalid inputs |

### Run 

> **Note:** `responses.jsonl` is the output file of running the baselines approaches

> **Note:** `--max-pairs` is the max input/output test pairs to use in mutation

```bash
python -m spec_harness.eval_llm_response \
  --benchmark_path benchmarks/formalbench/fb.json \
  --llm_response_path path/to/responses.jsonl \
  --openjml openjml \
  --output spec_harness_results \
  --threads 8 \
  --max-pairs 5
  --verbose
```