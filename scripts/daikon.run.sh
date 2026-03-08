python -m baselines.daikon.run \
    --name sgb_run \
    --input /home/mdrh/code/specsyns/benchmarks/specgenbench/sgb.json \
    --output /home/mdrh/experiments/output \
    --openjml_timeout 300 \
    --daikon_timeout 600 \
    --threads 2 \
    --verbose