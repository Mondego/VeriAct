python -m optimizer.prompt_optimizer \
  --formalbench_path /home/mdrh/code/specsyns/benchmarks/formalbench/fb.json\
  --specgenbench_path /home/mdrh/code/specsyns/benchmarks/specgenbench/sgb.json \
  --optimizers gepa \
  -- best_seed specgen \
  --log_dir /home/mdrh/experiments/output_optimizer/optimizer_logs \
  --output_dir /home/mdrh/experiments/output_optimizer/optimizer_results