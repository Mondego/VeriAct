# Benchmarks

We use two established benchmarks for evaluation. 

- [SpecGenBench](./specgenbench/): Contains 120 Java method tasks
- [FormalBench](./formalbench/): Contains 700 Java method tasks


We normalize all benchmark tasks by renaming class names to
`Solution` and method names to `solve()`. We also extend each task
with 100–200 additional test pairs generated through randomly
guided automated test generation.