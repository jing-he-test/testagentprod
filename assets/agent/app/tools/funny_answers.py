"""
Easter-egg / funny answers for the agent.

Exposes a LangChain tool so the LLM can decide when to use it.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool
from tools.common import (
    get_user_from_context,
    # Re-export for backward compatibility with main.py middleware
    set_current_jwt_token,
    get_current_jwt_token,
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = ["funny_answer", "set_current_jwt_token", "get_current_jwt_token"]


@tool
async def funny_answer(question: str) -> str:
    """Return a funny or entertaining answer for light-hearted questions.

    Use this tool when the user asks playful questions such as who the best
    developer is, or when CAD will be delivered.

    Args:
        question: The user's question.
    """
    lower = question.lower()

    if "best developer" in lower:
        user = get_user_from_context() or "you"
        return f"The best developer is **{user}** — obviously."

    if "when" in lower and "cad" in lower and any(
        w in lower for w in ("deliver", "ship", "release", "launch")
    ):
        return "CAD will be delivered **just in time**."

    return "I don't have a funny answer for that one, but I tried!"
