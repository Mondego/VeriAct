# VeriAct: Beyond Verifiability 

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2604.01193-b31b1b.svg)](https://arxiv.org/pdf/2604.00280)
[![License](https://img.shields.io/badge/License-GNU_GPL_v3-blue)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)

### VeriAct: Beyond Verifiability -- Agentic Synthesis of Correct and Complete Formal Specifications
Md Rakib Hossain Misu\*, Iris Ma, Cristina V. Lopes
</div>


## ✨ Overview

This repository contains the implementation of our paper's methodology
> **VeriAct: Beyond Verifiability -- Agentic Synthesis of Correct and Complete Formal Specifications**

### 💎 Key Components

- **[baselines](./baselines/)** — implementation/execution scripts of classical (Daikon, Houdini) vs. prompt-based (SpecGen, AutoSpec, FormalBench) approaches.
- **[optimizer](./optimizer/)** — uses structured OpenJML feedback to iteratively refine prompts.
- **[spec_harness](./spec_harness/)** — evaluates correctness and completeness of verifier-accepted specifications beyond syntactic verification.
- **[veriact](./veriact/)** — an agentic loop that combines code execution, OpenJML verification, and Spec-Harness feedback to synthesize formal specifications.
- **[benchmarks](./benchmarks/)** — Two normalized benchmarks are used across experiments.

---
For full details, see the [paper](https://arxiv.org/pdf/2604.00280).


## 🚀 Getting Started

### 💻 Prerequisites

- Python >= 3.10
- [OpenJML](https://www.openjml.org/downloads/) — must be installed and available in `PATH` as `openjml`
- Check `openjml --version`

### ⏳ Install

```bash
git clone https://github.com/Mondego/VeriAct.git
cd VeriAct
uv sync --all-extras
source .venv/bin/activate
```
### 🔑 API Keys

Create a `.env` file in [config](./config/) with the keys for the models you intend to use:

<details>
<summary> API Keys</summary>

```bash
OPENAI_API_KEY=...       # GPT-4o 
ANTHROPIC_API_KEY=...    # Claude models
GOOGLE_API_KEY=...       # Gemini models
DEEPSEEK_API_KEY=...     # DeepSeek API
MISTRAL_API_KEY=...      # Mistral API
VLLM_API_KEY=...         # Local vLLM server
VLLM_API_BASE=...        # e.g. http://localhost:8000/v1
```
> **Note:** 
</details>


### 💣  Run Command

<details>
<summary>Run Daikon</summary>

> **Note:** Install [Daikon](https://plse.cs.washington.edu/daikon/download/doc/daikon.html#Installing-Daikon) and configure required jars in the `PATH` 

```bash
python -m baselines.daikon.run \
    --name <experiment_name> \
    --input <path/to/benchmark.json> \
    --output <output_dir> \
    --openjml_timeout 300 \
    --daikon_timeout 600 \
    --threads 4 \
    --verbose
```

</details>

<details>
<summary>Run Houdini</summary>

> **Note:** java version: [`1.6.0_21`](https://www.oracle.com/java/technologies/javase/6u21.html) requires to run Houdini 

```bash
python -m baselines.houdini.run \
    --name <experiment_name> \
    --input <path/to/benchmark.json> \
    --output <output_dir> \
    --openjml_timeout 300 \
    --threads 4 \
    --verbose
```


</details>

<details>
<summary>Run SpecGen</summary>

```bash
python -m baselines.specgen.run \
    --name <experiment_name> \
    --input <path/to/benchmark.json> \
    --output <output_dir> \
    --model gpt-4o \
    --temperature 0.7 \
    --prompt_type zero_shot \
    --max_iterations 10 \
    --openjml_timeout 300 \
    --threads 4 \
    --verbose
```

> **Note:** `prompt_type`: `zero_shot` | `two_shot` | `four_shot`
</details>


<details>
<summary>Run AutoSpec</summary>

```bash
python -m baselines.autospec.run \
    --name <experiment_name> \
    --input <path/to/benchmark.json> \
    --output <output_dir> \
    --model gpt-4o \
    --temperature 0.7 \
    --prompt_type zero_shot \
    --max_iterations 10 \
    --openjml_timeout 300 \
    --threads 4 \
    --verbose \
    --simplify
```

> **Note:** `prompt_type`: `zero_shot` | `two_shot` | `four_shot`
</details>


<details>
<summary>Run FormalBench</summary>

```bash
python -m baselines.formalbench.run \
    --name <experiment_name> \
    --input <path/to/benchmark.json> \
    --output <output_dir> \
    --model gpt-4o \
    --temperature 0.7 \
    --prompt_type zero_shot \
    --max_iterations 5 \
    --openjml_timeout 300 \
    --threads 4 \
    --verbose
```

> **Note:** `prompt_type`: `zero_shot` | `two_shot` | `zs_cot` | `fs_cot` | `fs_ltm`
</details>

<details>
<summary>Run Optimizer</summary>

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

> **Note:** `--best_seed`: `zero` | `cot` | `ltm`
</details>

<details>
<summary>Run Spec-Harness</summary>

```bash
python -m spec_harness.eval_llm_response \
  --benchmark_path benchmarks/formalbench/fb.json \
  --llm_response_path path/to/responses.jsonl \
  --openjml openjml \
  --output spec_harness_results \
  --threads 8 \
  --max-pairs 5 \
  --verbose
```

> **Note:** `responses.jsonl` is the output file of running the baselines approaches

> **Note:** `--max-pairs` is the max input/output test pairs to use in mutation
</details>

<details>

<summary>Run VeriAct</summary>


```bash
# Sequential (one task at a time)
python -m veriact.run_single \
    --benchmark <path/to/benchmark.json> \
    --model gpt-4o \
    --output-dir <output_dir> \
    --max-steps 12 \
    --planning_interval 3

# Parallel
python -m veriact.run_batch \
    --benchmark <path/to/benchmark.json> \
    --model gpt-4o \
    --threads 4 \
    --output-dir <output_dir> \
    --max-steps 12 \
    --planning_interval 3
```

> **Note:** Note
</details>


## 📁 Repository Structure

```
├── baselines               # All baselines implementation
│   ├── houdini             # Classic, template based approach 
│   ├── daikon              # Classic, execution trace based approach
│   ├── specgen             # Prompt based, spec mutation 
│   ├── autospec            # Prompt based, program decomposing and static analysis
│   ├── formalbench         # Prompt based, advance prompting with repair guidance
├── benchmarks              # Normalized benchamrk datasets
│   └── specgenbench        # 120 tasks from Leetcode and JML examples
│   ├── formalbench         # 662 tasks from FormalBench and MBJP
├── config                  # API keys in .env file
├── optimizer               # Prompt optimizer implementation
├── spec_harness            # Spec-Harness implementation
├── veriact                 # VeriAct agent implementation
├── scripts                 # CLI run scripts for all baselines, veriact
├── pyproject.toml
├── README.md

```



## 📝 Citation

```bibtex
@misc{misu2026veriactverifiabilityagentic,
      title={VeriAct: Beyond Verifiability -- Agentic Synthesis of Correct and Complete Formal Specifications}, 
      author={Md Rakib Hossain Misu and Iris Ma and Cristina V. Lopes},
      year={2026},
      eprint={2604.00280},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2604.00280}, 
}
```

## 📧 Contact
If you have any questions or find any issues, please contact us at [mdrh@uci.edu](mailto:mdrh@uci.edu)


## 📄 License
This repository is licensed under [GNU General Public License v3.0](LICENSE)
