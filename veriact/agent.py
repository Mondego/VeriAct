"""VeriActAgent: CODEACT mode entry point."""

from __future__ import annotations
import json
import logging
import os
from datetime import datetime

from veriact.codeact import CodeAgent
from veriact.default_tools import TaskCompletionTool
from veriact.data_types import Task, HARNESS_PASS_THRESHOLD
from veriact.file_utility import dump_json
from veriact.memory import ActionStep
from veriact.tools import get_veriact_tools
from veriact.utility import AgentMaxStepsError

logger = logging.getLogger(__name__)


class VeriActAgent:
    """Verification-Guided Agentic Framework for Formal Specification Synthesis.

    Args:
        model: Any veriact Model (OpenAIServerModel, AnthropicModel, GeminiModel, VLLMModel).
        openjml_path: Path to OpenJML binary.
        dataset_path: Path to the dataset used by the harness/retrieval tools.
        output_dir: Directory for outputs.
    """

    def __init__(
        self,
        model,
        openjml_path="openjml",
        dataset_path="../benchmarks/example/data.json",
        output_dir="veriact_outputs",
        planning_interval=5,
        max_steps=15,
        harness_threshold=HARNESS_PASS_THRESHOLD,
        _run_dir: str | None = None,
        **kwargs,
    ):
        self.model = model
        if _run_dir:
            # Use a pre-computed run directory (e.g. from batch mode)
            self.output_dir = _run_dir
        else:
            model_id = getattr(model, "model_id", "unknown") or "unknown"
            safe_model_id = model_id.replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(output_dir, f"veriact__{safe_model_id}__{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        self._tools = get_veriact_tools(
            openjml_path=openjml_path,
            dataset_path=dataset_path,
            output_dir=self.output_dir,
        )
        self._codeact_kwargs = {
            "planning_interval": planning_interval,
            "max_steps": max_steps,
            "harness_threshold": harness_threshold,
            **kwargs,
        }

    def run(self, task: Task) -> dict:
        agent = CodeAgent(
            tools=self._tools + [TaskCompletionTool()],
            model=self.model,
            **self._codeact_kwargs,
        )

        task_str = f"task_id: {task.task_id}\n\n" f"java_code:\n{task.code}\n\n"
        result = agent.run(task=task_str)
        _last_attempted_code = agent.get_last_jml_code()

        # Determine success: True only if the agent called task_complete
        # voluntarily (not forced by hitting max_steps) AND the last
        # run_spec_harness scores met the threshold.
        action_steps = [
            s
            for s in agent.memory.steps
            if isinstance(s, ActionStep) and s.step_number is not None
        ]
        hit_max_steps = any(
            isinstance(s.error, AgentMaxStepsError) for s in action_steps
        )
        harness_passed = self._check_harness_passed(
            action_steps, self._codeact_kwargs.get("harness_threshold", HARNESS_PASS_THRESHOLD)
        )
        success = result is not None and not hit_max_steps and harness_passed

        # Count only real agent steps (exclude the forced max-steps step)
        iterations = sum(
            1 for s in action_steps if not isinstance(s.error, AgentMaxStepsError)
        )

        trajectory = {
            "task_id": task.task_id,
            "success": success,
            "iterations": iterations,
            "agent_output": result,
            "agent_dict": agent.to_dict(),
            "_last_attempted_code": _last_attempted_code,
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
        """Return True if the last run_spec_harness call met the threshold."""
        for step in reversed(action_steps):
            for tool_out in reversed(step.tool_outputs or []):
                if tool_out.get("tool_name") != "run_spec_harness":
                    continue
                # Parse the harness output (JSON string with scores)
                raw = tool_out.get("output", "")
                try:
                    scores = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    continue
                # scores is {task_id: {metric: value, ...}}
                if isinstance(scores, dict):
                    for _tid, metrics in scores.items():
                        if not isinstance(metrics, dict):
                            continue
                        pc = metrics.get("post_correctness", 0.0)
                        pcm = metrics.get("post_completeness", 0.0)
                        if pc >= threshold and pcm >= threshold:
                            return True
                        return False  # found last harness call but didn't pass
        return False  # no harness call found

    def to_dict(self):
        return {
            "mode": "codeact",
            "model_id": getattr(self.model, "model_id", "?"),
            "tools": [t.name for t in self._tools],
        }
