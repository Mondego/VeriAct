"""
veriact — CLI-driven refactor of VeriAct.

Same verification-guided spec-synthesis loop, but the agent drives **command-line
tools** (verify, run_spec_harness, task_complete) in a think → run-CLI → observe
ReAct loop, instead of executing Python via CodeAct. Memory, monitoring, and
trajectory recording are preserved.

    from veriact import VeriActAgent, OpenAIServerModel
    agent = VeriActAgent(model=OpenAIServerModel(model_id="gpt-4o"))
    agent.run(task)
"""

from veriact.agent import VeriActAgent
from veriact.core.data_types import Task, HARNESS_PASS_THRESHOLD
from veriact.core.models import (
    OpenAIServerModel,
    AnthropicModel,
    GeminiModel,
    VLLMModel,
)

__version__ = "0.1.0"
