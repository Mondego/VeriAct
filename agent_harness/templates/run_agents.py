#!/usr/bin/env python3
"""Parent-level driver — fan out one fresh agent session per task directory.

Run this from the out-root produced by ``agent_harness.scaffold``. For each task
subdir (any immediate child containing ``AGENTS.md``) it launches a headless
coding-agent session with that subdir as the working directory, feeding the dir's
``AGENTS.md`` as the prompt. One isolated session per task, matching VeriAct's
per-task runs — so VeriAct / Claude Code / Codex are all driven identically.

This script is self-contained (stdlib only); it ships inside the out-root.

Default models
--------------
    --agent claude  ->  Claude Code on  claude-sonnet-4-6
    --agent codex   ->  Codex on the model from ~/.codex/config.toml (no -m passed)

Examples
--------
    # Claude Code (Sonnet 4.6 by default), headless, auto-approve:
    python run_agents.py --agent claude --threads 4 --only-missing
    #   expands to: claude -p {agents_md} --dangerously-skip-permissions --model claude-sonnet-4-6

    # Codex (model from ~/.codex/config.toml, no bwrap sandbox):
    python run_agents.py --agent codex --threads 4 --only-missing
    #   expands to: codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox {agents_md}

    # Pin a model / override any flag with --cmd:
    python run_agents.py --cmd 'claude -p {agents_md} --dangerously-skip-permissions --model claude-opus-4-8'
    python run_agents.py --cmd 'codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox --model gpt-5-codex {agents_md}'
    python run_agents.py --cmd 'mytool run --prompt-file {agents_md_path}' --timeout 1200
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))

# Built-in agent command templates. Tokens are replaced per task:
#   {agents_md}       -> full text of the dir's AGENTS.md (single argv token)
#   {agents_md_path}  -> absolute path to AGENTS.md
#   {task_dir}        -> absolute path to the task directory
#   {task_id}         -> directory name
#
# These run headless and must edit files + run the *.sh tools WITHOUT prompting,
# so they include each CLI's "auto-approve" flag and a default model. Only run on an
# isolated/sandboxed host. Override flags or the model with --cmd if your CLI version
# differs.
BUILTIN_AGENTS = {
    "claude": "claude -p {agents_md} --dangerously-skip-permissions --model claude-sonnet-4-6",
    # Codex runs WITHOUT its bubblewrap sandbox: in many containers/VMs bwrap cannot
    # create its loopback ("RTM_NEWADDR: Operation not permitted"), which makes every
    # file edit / command fail. --dangerously-bypass-approvals-and-sandbox avoids bwrap
    # entirely (safe only on an already-isolated host, like the claude default).
    # No -m: the model comes from your Codex config (~/.codex/config.toml). Override
    # per run with --cmd '... -m <id> ...' if needed. Note gpt-4o requires an OpenAI
    # API-key login (a ChatGPT-account login rejects it).
    "codex": (
        "codex exec --skip-git-repo-check "
        "--dangerously-bypass-approvals-and-sandbox {agents_md}"
    ),
}


def find_task_dirs() -> list[str]:
    dirs = []
    for name in sorted(os.listdir(ROOT)):
        d = os.path.join(ROOT, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "AGENTS.md")):
            dirs.append(d)
    return dirs


def build_command(template: str, task_dir: str) -> list[str]:
    agents_md_path = os.path.join(task_dir, "AGENTS.md")
    with open(agents_md_path) as fh:
        agents_md = fh.read()
    tokens = shlex.split(template)
    repl = {
        "{agents_md}": agents_md,
        "{agents_md_path}": agents_md_path,
        "{task_dir}": task_dir,
        "{task_id}": os.path.basename(task_dir),
    }
    out = []
    for tok in tokens:
        for k, v in repl.items():
            tok = tok.replace(k, v)
        out.append(tok)
    return out


def run_one(template: str, task_dir: str, timeout: int | None) -> tuple[str, int, str]:
    task_id = os.path.basename(task_dir)
    cmd = build_command(template, task_dir)
    log_path = os.path.join(task_dir, "harness", "agent_run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    try:
        with open(log_path, "w") as log:
            proc = subprocess.run(
                cmd,
                cwd=task_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        return task_id, proc.returncode, log_path
    except subprocess.TimeoutExpired:
        return task_id, -1, log_path + " (timeout)"
    except FileNotFoundError as exc:
        return task_id, -2, f"command not found: {exc}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Drive an agent over scaffolded task dirs.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--agent", choices=sorted(BUILTIN_AGENTS), help="built-in agent template")
    g.add_argument("--cmd", help="custom command template (see token list in --help)")
    p.add_argument("--task-ids", help="file with task ids to run (one per line)")
    p.add_argument("--threads", type=int, default=1, help="parallel sessions (default: 1)")
    p.add_argument("--timeout", type=int, default=None, help="per-task timeout seconds")
    p.add_argument(
        "--only-missing",
        action="store_true",
        help="skip dirs that already have submission.json",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    template = args.cmd or BUILTIN_AGENTS[args.agent]

    task_dirs = find_task_dirs()
    if args.task_ids:
        with open(args.task_ids) as fh:
            wanted = {ln.strip() for ln in fh if ln.strip()}
        task_dirs = [d for d in task_dirs if os.path.basename(d) in wanted]
    if args.only_missing:
        task_dirs = [
            d for d in task_dirs if not os.path.exists(os.path.join(d, "submission.json"))
        ]

    if not task_dirs:
        print("No task dirs to run.", file=sys.stderr)
        return 1

    print(f"Running '{template}' over {len(task_dirs)} task(s), threads={args.threads}")
    results: list[tuple[str, int, str]] = []
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futs = {
            pool.submit(run_one, template, d, args.timeout): d for d in task_dirs
        }
        for fut in as_completed(futs):
            task_id, rc, log = fut.result()
            status = "ok" if rc == 0 else f"rc={rc}"
            print(f"  [{status:>6}] {task_id}  ({log})")
            results.append((task_id, rc, log))

    ok = sum(1 for _, rc, _ in results if rc == 0)
    print(f"Done: {ok}/{len(results)} sessions exited 0. "
          f"Next: python collect.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
