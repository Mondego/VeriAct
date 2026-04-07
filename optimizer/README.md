# GEPA-Prompt Optimization for Formal Specification Synthesis

**GEPA Optimizer** — uses structured OpenJML feedback to iteratively refine prompts.


### Run

```bash
python -m optimizer.prompt_optimizer \
    --formalbench_path <path/to/benchmarks/formalbench/fb.json> \
    --specgenbench_path <path/to/benchmarks/specgenbench/sgb.json> \
    --optimizers gepa \
    --best_seed zero \           
    --model openai/gpt-4o \
    --reflection_model openai/gpt-4o \
    --log_dir optimizer_logs \
    --output_dir optimizer_results \
    --openjml_output_dir openjml_output
```