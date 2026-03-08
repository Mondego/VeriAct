python -m baselines.houdini.run \
    --name sgb_run \
    --input /home/mdrh/code/specsyns/benchmarks/specgenbench/sgb.json \
    --output /home/mdrh/experiments/exp-results/houdini \
    --openjml_timeout 300 \
    --threads 8 \
    --verbose