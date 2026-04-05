# VeriAct
Verification-guided correct & complete formal specification synthesis. The agent iteratively writes JML specifications, verifies them with OpenJML, and evaluates correctness/completeness with Spec-Harness until the specifications pass both checks.

## How It Works

```
run_single.py / run_batch.py
        │
        ▼
  VeriActAgent          ← loads tools, wraps CodeAgent
        │
        ▼
   CodeAgent            ← ReAct loop: think → execute Python code → observe
        │
   ┌────┴──────────────────────────┐
   ▼                               ▼
verify_with_openjml         run_spec_harness
   (verifier_tool.py)         (harness_tool.py)
        │                          │
   OpenJML (ESC)           Spec-Harness evaluation
                           post/pre correctness & completeness
```

Each step the agent outputs `{"thought": "...", "code": "..."}`. The code runs in a sandboxed Python environment and calls tools directly. The agent succeeds when `post_correctness ≥ 0.5` AND `post_completeness ≥ 0.5` and calls `task_complete`.

## Module Reference

| File | What it does |
|------|-------------|
| `agent.py` | `VeriActAgent` — top-level entry point; wraps `CodeAgent`, checks harness threshold, writes output |
| `codeact.py` | `CodeAgent` / `MultiStepAgent` — ReAct loop; executes agent-generated Python code in a sandboxed namespace |
| `data_types.py` | `Task`, `TestCase`, `HARNESS_PASS_THRESHOLD = 0.5` |
| `models.py` | LLM backends: `OpenAIServerModel`, `AnthropicModel`, `GeminiModel`, `VLLMModel` |
| `tools.py` | `get_veriact_tools()` — returns the three agent tools (see below) |
| `tools_base.py` | `Tool` base class with input validation and type checking |
| `verifier_tool.py` | `verify_with_openjml()` — writes code to disk, runs OpenJML ESC, parses and classifies errors |
| `harness_tool.py` | `evaluate_problem()` — runs Spec-Harness, returns the four metric scores |
| `memory.py` | `ActionStep`, `AgentMemory` — records each agent step for logging and replay |
| `agent_types.py` | `AgentText` type wrapper for agent I/O |
| `prompts/veriact_prompt.yaml` | System prompt: role, tool descriptions, workflow, timeout strategy |

## Agent Tools

| Tool name | Inputs | Returns |
|-----------|--------|---------|
| `verify_with_openjml` | `jml_annotated_code: str` | `{verified, return_code, errors, raw_output}` |
| `analyze_openjml_errors` | `openjml_log: str` | `{failure_modes, repair_hints, summary}` |
| `run_spec_harness` | `task_id: str`, `jml_annotated_code: str` | `{post_correctness, post_completeness, pre_correctness, pre_completeness}` |
| `task_complete` | — | signals successful completion |


## Usage

```bash
cd VeriAct

# Sequential — one task at a time
python -m veriact.run_single.py \
    --benchmark benchmarks/specgenbench/sgb.json \
    --model gpt-4o \
    --output-dir <output_dir> \
    --max-steps 15 \
    --planning_interval 3


# Parallel
python -m veriact.run_batch.py \
    --benchmark benchmarks/specgenbench/sgb.json \
    --model gpt-4o \
    --threads 4 \
    --output-dir <output_dir> \
    --max-steps 15
```

| Flag | Default | Description |
|------|---------|-------------|
| `--benchmark` | required | Dataset JSON/JSONL path |
| `--model` | `gpt-4o` | LLM model ID — provider auto-detected from prefix (`gpt-`/`claude-`/`gemini-`) |
| `--output-dir` | `veriact_outputs` | Output directory |
| `--openjml-path` | `openjml` | Path to OpenJML binary |
| `--max-steps` | 15 | Max agent iterations per task |
| `--planning_interval` | 5 | Steps between planning phases |
| `--threads` | 4 | Parallel workers (`run_batch.py` only) |
| `--task-ids` | — | File with task IDs to filter (`run_batch.py` only) |

Set `VLLM_API_BASE` environment variable to use a local vLLM server.

## Output

Each run produces:

```
<output-dir>/
├── trajectories.jsonl         # One JSON object per task
├── trajectories/
│   └── <task_id>_veriact_trajectory.json
└── failed_tasks.json          # Tasks that errored (if any)
```

Each entry in `trajectories.jsonl`:

```json
{
  "task_id": "...",
  "success": true,
  "iterations": 7,
  "agent_output": "...",
  "agent_dict": { ... },
  "_last_attempted_code": "..."
}
```
