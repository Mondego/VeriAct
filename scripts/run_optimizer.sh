python -m optimizer.prompt_optimizer \
  --formalbench_path /home/mdrh/code/VeriAct/benchmarks/formalbench/fb.json\
  --specgenbench_path /home/mdrh/code/VeriAct/benchmarks/specgenbench/sgb.json \
  --optimizers gepa \
  -- best_seed zero \
  --log_dir /home/mdrh/experiments/output_optimizer/optimizer_logs_zero \
  --output_dir /home/mdrh/experiments/output_optimizer/optimizer_results_zero \
  --openjml_output_dir /home/mdrh/experiments/output_optimizer/openjml_output_zero