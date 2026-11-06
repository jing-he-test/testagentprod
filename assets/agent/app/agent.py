"""
CAD Example Agent using LangChain with LiteLLM.

This module demonstrates Application Foundation telemetry features:
- context_overlay for custom span creation
- chat_span for LLM call tracing
- Custom attributes for business context
- Span events for workflow milestones
- Token usage metrics recording
"""

# Standard library imports
import logging
import warnings
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# Third-party imports
from langchain_litellm import ChatLiteLLM
from rich.console import Console
from rich.markdown import Markdown

# Application Foundation telemetry imports
from sap_cloud_sdk.core.telemetry import (
    GenAIOperation,
    add_span_attribute,
    chat_span,
    context_overlay,
)
from tools.debugging import list_available_tools, list_ariba_events, list_products_from_s4
from tools.funny_answers import funny_answer

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant built with Application Foundation.

Your capabilities:
- Answer questions and provide helpful information
- List available MCP tools when asked about capabilities
- Retrieve product data from SAP S/4HANA
- Retrieve sourcing events from Ariba
- Give funny and entertaining responses for light-hearted questions

Be concise, helpful, and friendly in your responses."""

TOOLS = [list_available_tools, list_products_from_s4, list_ariba_events, funny_answer]

# Map tool name → callable for dispatch
_TOOL_MAP = {t.name: t for t in TOOLS}


@dataclass
class AgentResponse:
    """Response from the agent."""

    status: Literal["input_required", "completed", "error"]
    message: str


class CADExampleAgent:
    """
    CAD Example Agent using LangChain with LiteLLM.

    Uses an agentic loop: the LLM decides which tools to call, tools are
    executed, results are fed back, and the loop continues until the model
    produces a plain text response with no tool calls.
    """

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]  # noqa: RUF012

    def __init__(self):
        """Initialize the CAD Example Agent."""
        base_llm = ChatLiteLLM(
            model="sap/anthropic--claude-4.5-sonnet",
        )
        self.llm = base_llm.bind_tools(TOOLS)
        self.console = Console()

    async def _run_tool(self, tool_call: dict) -> ToolMessage:
        """Execute a single tool call and return a ToolMessage."""
        name = tool_call["name"]
        args = tool_call["args"]
        tool_call_id = tool_call["id"]

        logger.info("Executing tool: %s, args: %s", name, args)

        tool_fn = _TOOL_MAP.get(name)
        if tool_fn is None:
            result = f"Unknown tool: {name}"
        else:
            try:
                result = await tool_fn.ainvoke(args)
            except Exception as e:
                logger.exception("Tool %s raised an exception", name)
                result = f"Error running {name}: {e}"

        return ToolMessage(content=str(result), tool_call_id=tool_call_id)

    async def _execute_with_telemetry(self, query: str, context_id: str) -> str:
        """
        Run the agentic loop with comprehensive telemetry tracking.

        The loop:
          1. Send messages to the LLM.
          2. If the response contains tool calls, execute them and append results.
          3. Repeat until the LLM returns a plain text response.
        """
        with context_overlay(
            GenAIOperation.INVOKE_AGENT,
            attributes={
                "user.id": "u-demo-123",
                "tenant.id": "tenant-appfnd",
                "conversation.id": context_id,
                "agent.name": "CADExampleAgent",
                "agent.framework": "langchain",
                "agent.model": "sap/anthropic--claude-4.5-sonnet",
                "query.length": len(query),
                "query.word_count": len(query.split()),
            },
        ) as agent_span:
            agent_span.add_event(
                "agent_invocation_started",
                {"query_preview": query[:100] + "..." if len(query) > 100 else query},
            )

            messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]
            total_input_tokens = 0
            total_output_tokens = 0
            iteration = 0

            try:
                while True:
                    iteration += 1
                    logger.info("Agentic loop iteration %d", iteration)

                    with chat_span(
                        model="sap/anthropic--claude-4.5-sonnet",
                        provider="anthropic",
                        conversation_id=context_id,
                        server_address="api.anthropic.com",
                        attributes={
                            "gen_ai.request.temperature": 0.7,
                            "gen_ai.request.max_tokens": 4096,
                            "agent.loop.iteration": iteration,
                        },
                    ) as llm_span:
                        llm_span.add_event("model_invocation_started")

                        response = await self.llm.ainvoke(messages)
                        if isinstance(response.content, str):
                            full_response = response.content
                        else:
                            full_response = "".join(
                                block.get("text", "")
                                for block in response.content
                                if isinstance(block, dict) and block.get("type") == "text"
                            )
                        tool_calls = response.tool_calls if hasattr(response, "tool_calls") and response.tool_calls else []
                        token_usage = response.usage_metadata if hasattr(response, "usage_metadata") else None

                        # Keep the AIMessage for history
                        ai_message = AIMessage(
                            content=full_response,
                            tool_calls=tool_calls,
                        )

                        llm_span.add_event(
                            "model_response_received",
                            {
                                "response_length": len(full_response),
                                "tool_calls_count": len(tool_calls),
                            },
                        )
                        add_span_attribute("gen_ai.response.finish_reason", "complete")

                        if token_usage:
                            in_tok = token_usage.get("input_tokens", 0)
                            out_tok = token_usage.get("output_tokens", 0)
                            total_input_tokens += in_tok
                            total_output_tokens += out_tok
                            add_span_attribute("gen_ai.usage.input_tokens", in_tok)
                            add_span_attribute("gen_ai.usage.output_tokens", out_tok)

                    messages.append(ai_message)

                    # No tool calls → final answer
                    if not tool_calls:
                        break

                    # Execute all requested tools and append results
                    agent_span.add_event("tools_execution_started", {"count": len(tool_calls)})
                    with context_overlay(
                        GenAIOperation.EXECUTE_TOOL,
                        attributes={"tool.count": len(tool_calls)},
                    ):
                        for tc in tool_calls:
                            tool_msg = await self._run_tool(tc)
                            messages.append(tool_msg)
                    agent_span.add_event("tools_execution_completed", {"count": len(tool_calls)})

                agent_span.add_event(
                    "agent_invocation_completed",
                    {
                        "output_length": len(full_response),
                        "output_word_count": len(full_response.split()),
                        "total_input_tokens": total_input_tokens,
                        "total_output_tokens": total_output_tokens,
                        "loop_iterations": iteration,
                    },
                )

                return full_response

            except Exception as e:
                agent_span.add_event(
                    "error_occurred",
                    {"error.type": type(e).__name__, "error.message": str(e)},
                )
                add_span_attribute("error", True)
                add_span_attribute("error.type", type(e).__name__)
                logger.exception("Error during agentic loop")
                raise

    async def stream(self, query: str, context_id: str) -> AsyncGenerator[dict, None]:
        """
        Stream responses from the agent.

        The LLM decides which tools (if any) to invoke. Tool results are fed
        back into the conversation until the model produces a final answer.

        Args:
            query: The user's question or request
            context_id: Unique identifier for the conversation context

        Yields:
            Status updates and final response
        """
        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "content": "Processing your request...",
        }

        try:
            response_content = await self._execute_with_telemetry(query, context_id)

            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": response_content,
            }

        except Exception as e:
            logger.exception("Error in agent stream")

            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Error processing request: {e!s}",
            }

    def invoke(self, query: str, context_id: str) -> AgentResponse:
        """
        Synchronous invocation of the agent with telemetry.

        Args:
            query: The user's question or request
            context_id: Unique identifier for the conversation context

        Returns:
            AgentResponse with status and message
        """
        import asyncio  # noqa: PLC0415

        async def _run():
            result_content = None
            async for item in self.stream(query, context_id):
                if item["is_task_complete"]:
                    result_content = item["content"]
            return result_content

        try:
            result_content = asyncio.run(_run())
            return AgentResponse(status="completed", message=result_content)
        except Exception as e:
            return AgentResponse(status="error", message=f"Error: {e!s}")

    def print_response(self, response: str) -> None:
        """Print a response with markdown formatting."""
        self.console.print(Markdown(response))
