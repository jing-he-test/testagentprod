"""
Unit tests for the CAD example agent.

Run from the agent directory with:
    PYTHONPATH=app python3 -m pytest app/test_agent.py -v

No live credentials required — LLM and tool calls are mocked.
"""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock
from langchain_core.messages import AIMessage, ToolMessage

# Minimal env vars required for module-level imports in agent.py / tools/
os.environ.setdefault("AICORE_AUTH_URL", "https://fake.auth/token")
os.environ.setdefault("AICORE_BASE_URL", "https://fake.aicore")
os.environ.setdefault("AICORE_CLIENT_ID", "fake-id")
os.environ.setdefault("AICORE_CLIENT_SECRET", "fake-secret")
os.environ.setdefault("AICORE_RESOURCE_GROUP", "default")
os.environ.setdefault("INTEGRATION_AUTH_URL", "https://fake.ias/token")
os.environ.setdefault("INTEGRATION_CLIENT_ID", "fake-client")
os.environ.setdefault("INTEGRATION_GATEWAY_URL", "https://fake.gateway")
os.environ.setdefault("INTEGRATION_DEPENDENCIES", "[]")

from agent import CADExampleAgent  # noqa: E402 (env vars must be set first)


def _make_agent(llm_responses: list[AIMessage]) -> CADExampleAgent:
    """Return a CADExampleAgent whose LLM returns the given responses in order."""
    agent = CADExampleAgent()
    call_count = 0
    responses = llm_responses

    async def fake_ainvoke(messages, **kw):
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return resp

    agent.llm = AsyncMock()
    agent.llm.ainvoke = fake_ainvoke
    return agent


async def _collect(agent: CADExampleAgent, query: str) -> list[dict]:
    results = []
    async for chunk in agent.stream(query, "test-ctx"):
        results.append(chunk)
    return results


@pytest.mark.asyncio
async def test_simple_text_answer():
    """LLM returns a plain text answer — no tool calls."""
    agent = _make_agent([AIMessage(content="42 — cannot commit to a date.")])
    results = await _collect(agent, "when will cad be delivered?")

    assert results[-1]["is_task_complete"] is True
    assert "42" in results[-1]["content"]


@pytest.mark.asyncio
async def test_list_content_text_extraction():
    """When response.content is a list (text + tool_use blocks), text is extracted correctly."""
    agent = _make_agent([AIMessage(content=[
        {"type": "text", "text": "CAD will be delivered just in time."},
        {"type": "tool_use", "id": "tc-1", "name": "funny_answer", "input": {}},
    ])])
    results = await _collect(agent, "when will cad be delivered?")

    assert results[-1]["is_task_complete"] is True
    assert "just in time" in results[-1]["content"]


@pytest.mark.asyncio
async def test_tool_call_then_text_answer():
    """LLM requests a tool call on iteration 1, returns text on iteration 2."""
    tool_call = {"name": "funny_answer", "args": {"question": "when?"}, "id": "tc-1"}
    agent = _make_agent([
        AIMessage(content="", tool_calls=[tool_call]),
        AIMessage(content="Soon — trust the process."),
    ])
    # _run_tool is a plain method — freely replaceable
    agent._run_tool = AsyncMock(return_value=ToolMessage(content="Soon!", tool_call_id="tc-1"))

    results = await _collect(agent, "when?")

    assert results[-1]["is_task_complete"] is True
    assert "Soon" in results[-1]["content"]
    assert agent._run_tool.call_count == 1


@pytest.mark.asyncio
async def test_llm_error_returns_graceful_response():
    """An exception from the LLM is caught and returned as an error message."""
    agent = CADExampleAgent()
    agent.llm = AsyncMock()
    agent.llm.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

    results = await _collect(agent, "hello")

    assert results[-1]["is_task_complete"] is True
    assert "boom" in results[-1]["content"]


@pytest.mark.asyncio
async def test_no_streaming_attributes_on_llm():
    """ChatLiteLLM must NOT be initialized with streaming=True.

    The agent facade never yields tokens — using the streaming path
    triggers the LiteLLM/Pydantic MockValSer crash on tool-call responses.
    """
    agent = CADExampleAgent()
    # After bind_tools() the llm is a _ChatModelBinding; the underlying
    # ChatLiteLLM is at .bound
    base_llm = getattr(agent.llm, "bound", agent.llm)
    assert not getattr(base_llm, "streaming", False), (
        "ChatLiteLLM must not use streaming=True — see fix/agent-mockvalser-streaming"
    )
