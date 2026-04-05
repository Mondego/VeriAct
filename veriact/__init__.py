"""
VeriAct — Verification-Guided Agentic Framework for Formal Specification Synthesis.

Usage:
    from veriact import VeriActAgent, OpenAIServerModel

    model = OpenAIServerModel(model_id="gpt-4o")
    agent = VeriActAgent(model=model)
    result = agent.run(task)
"""

from veriact.agent import VeriActAgent
from veriact.data_types import Task, HARNESS_PASS_THRESHOLD
from veriact.models import (
    OpenAIServerModel,
    AnthropicModel,
    GeminiModel,
    VLLMModel,
)

__version__ = "0.1.0"
