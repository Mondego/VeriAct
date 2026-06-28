"""VeriActAgent — CLI-ReAct entry point (drop-in replacement for VeriActAgent).

Per task it materializes a small workspace (Solution.java placeholder, Test.java,
harness/task.json), runs the CLIAgent ReAct loop (which drives the tool CLI as
subprocesses), and records the same trajectory / memory / monitoring artifacts as
the original VeriAct.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

from veriact.agent.cli_agent import CLIAgent
from veriact.core.data_types import HARNESS_PASS_THRESHOLD, Task
from veriact.core.file_utility import dump_json
from veriact.core.memory import ActionStep
from veriact.tools.descriptors import RunHarnessTool, VerifyTool
from veriact.core.utility import AgentMaxStepsError

logger = logging.getLogger(__name__)


def _task_to_record(task: Task) -> dict:
    """Serialize a Task to the dict shape harness_tool.Task.from_dict expects."""
    return {
        "task_id": task.task_id,
        "code": task.code,
        "class_name": task.class_name,
        "test_name": task.test_name,
        "javadoc": task.javadoc,
        "category": task.category,
        "origin_id": task.origin_id,
        "test_code": task.test_code,
        "test_inputs": [{"input": tc.input, "output": tc.output} for tc in task.test_inputs],
        "generated_test_cases": [
            {"input": tc.input, "output": tc.output} for tc in task.generated_test_cases
        ],
    }


class VeriActAgent:
    """CLI-driven verification-guided spec synthesis agent."""

    def __init__(
        self,
        model,
        openjml_path="openjml",
        dataset_path="",
        output_dir="veriact_outputs",
        planning_interval=5,
        max_steps=15,
        harness_threshold=HARNESS_PASS_THRESHOLD,
        no_harness: bool = False,
        _run_dir: str | None = None,
        **kwargs,
    ):
        self.model = model
        self.openjml_path = openjml_path
        self.dataset_path = dataset_path
        self.max_steps = max_steps
        self.planning_interval = planning_interval
        self.harness_threshold = harness_threshold
        self.no_harness = no_harness
        self._kwargs = kwargs

        if _run_dir:
            self.output_dir = _run_dir
        else:
            model_id = getattr(model, "model_id", "unknown") or "unknown"
            safe_model_id = model_id.replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(
                output_dir, f"veriact__{safe_model_id}__{timestamp}"
            )
        os.makedirs(self.output_dir, exist_ok=True)

    def _setup_workspace(self, task: Task) -> tuple[str, str]:
        safe_id = task.task_id.replace("/", "_").replace("\\", "_")
        ws = os.path.join(self.output_dir, "workspaces", safe_id)
        os.makedirs(os.path.join(ws, "harness"), exist_ok=True)
        # Test.java for the agent / harness; Solution.java seeded with the bare code.
        with open(os.path.join(ws, "Test.java"), "w") as fh:
            fh.write(task.test_code or "")
        with open(os.path.join(ws, "Solution.java"), "w") as fh:
            fh.write(task.code or "")
        task_json = os.path.join(ws, "harness", "task.json")
        with open(task_json, "w") as fh:
            json.dump(_task_to_record(task), fh, indent=2)
        return ws, task_json

    def run(self, task: Task) -> dict:
        workspace, task_json = self._setup_workspace(task)

        # Ablation: no-harness arm exposes only verify (+ task_complete) and uses
        # the harness-free prompt; success = verification passes.
        tools = [VerifyTool()] if self.no_harness else [VerifyTool(), RunHarnessTool()]
        prompt_name = (
            "veriact_cli_no_harness_prompt.yaml"
            if self.no_harness
            else "veriact_cli_prompt.yaml"
        )
        agent = CLIAgent(
            tools=tools,
            model=self.model,
            task_id=task.task_id,
            workspace_dir=workspace,
            task_json=task_json,
            openjml_path=self.openjml_path,
            max_steps=self.max_steps,
            planning_interval=self.planning_interval,
            harness_threshold=self.harness_threshold,
            prompt_name=prompt_name,
            **self._kwargs,
        )

        task_str = f"task_id: {task.task_id}\n\njava_code:\n{task.code}\n\n"
        result = agent.run(task=task_str)
        last_code = agent.get_last_jml_code()

        action_steps = [
            s for s in agent.memory.steps
            if isinstance(s, ActionStep) and s.step_number is not None
        ]
        hit_max_steps = any(isinstance(s.error, AgentMaxStepsError) for s in action_steps)
        if self.no_harness:
            passed = self._check_verified(action_steps)
        else:
            passed = self._check_harness_passed(action_steps, self.harness_threshold)
        success = result is not None and not hit_max_steps and passed
        iterations = sum(
            1 for s in action_steps if not isinstance(s.error, AgentMaxStepsError)
        )

        trajectory = {
            "task_id": task.task_id,
            "success": success,
            "iterations": iterations,
            "agent_output": result,
            "agent_dict": agent.to_dict(),
            "_last_attempted_code": last_code,
        }

        trajectories_dir = os.path.join(self.output_dir, "trajectories")
        os.makedirs(trajectories_dir, exist_ok=True)
        dump_json(
            trajectory,
            os.path.join(trajectories_dir, f"{task.task_id}_veriact_trajectory.json"),
        )
        return trajectory

    @staticmethod
    def _check_harness_passed(action_steps: list[ActionStep], threshold: float) -> bool:
        """True if the last run_spec_harness call met the threshold."""
        for step in reversed(action_steps):
            for tool_out in reversed(step.tool_outputs or []):
                if tool_out.get("tool_name") != "run_spec_harness":
                    continue
                raw = tool_out.get("output", "")
                try:
                    scores = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(scores, dict):
                    for _tid, metrics in scores.items():
                        if not isinstance(metrics, dict):
                            continue
                        pc = metrics.get("post_correctness", 0.0)
                        pcm = metrics.get("post_completeness", 0.0)
                        return pc >= threshold and pcm >= threshold
        return False

    @staticmethod
    def _check_verified(action_steps: list[ActionStep]) -> bool:
        """No-harness success: True if the last verify call reported verified=True."""
        for step in reversed(action_steps):
            for tool_out in reversed(step.tool_outputs or []):
                if tool_out.get("tool_name") != "verify":
                    continue
                raw = tool_out.get("output", "")
                try:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(data, dict):
                    return bool(data.get("verified"))
        return False

    def to_dict(self):
        tools = ["verify", "task_complete"] if self.no_harness else [
            "verify", "run_spec_harness", "task_complete"
        ]
        return {
            "mode": "cli-react" + ("-no-harness" if self.no_harness else ""),
            "model_id": getattr(self.model, "model_id", "?"),
            "tools": tools,
        }
