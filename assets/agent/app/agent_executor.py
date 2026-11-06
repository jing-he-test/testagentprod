"""
A2A Agent Executor for the CAD Example Agent.
Handles A2A protocol integration and task management.
"""

import logging
import time

from a2a.server.agent_execution import AgentExecutor as BaseAgentExecutor
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, Task, TaskState, TaskStatus
from a2a.utils.errors import A2AServerError, MethodNotImplementedError
from agent import CADExampleAgent

# OpenTelemetry imports
from opentelemetry import trace

# Application Foundation telemetry imports
from sap_cloud_sdk.core.telemetry import (
    Module,
    add_span_attribute,
    invoke_agent_span,
    record_error_metric,
    record_request_metric,
    set_tenant_id,
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentExecutor(BaseAgentExecutor):
    """A2A executor that bridges the A2A protocol with the CADExampleAgent."""

    def __init__(self):
        """Initialize the agent executor with CADExampleAgent."""
        self.agent = CADExampleAgent()

    def _extract_tenant_id(self, context: RequestContext) -> str:
        """Extract tenant ID from the request context."""
        ctx_id = context.context_id or "unknown"
        tenant_id = f"tenant-{ctx_id[:8]}"

        logger.info(f"Extracted tenant ID: {tenant_id}")
        return tenant_id

    def _log_structured(
        self,
        level: int,
        message: str,
        task_id: str | None = None,
        context_id: str | None = None,
        tenant_id: str | None = None,
        status: str | None = None,
        include_trace: bool = True,
        **extra_attrs,
    ):
        """Log with structured attributes and trace correlation."""

        log_attrs = {}

        if task_id:
            log_attrs["a2a.task.id"] = task_id
        if context_id:
            log_attrs["a2a.context.id"] = context_id
        if tenant_id:
            log_attrs["tenant.id"] = tenant_id
        if status:
            log_attrs["event.status"] = status

        if include_trace:
            span = trace.get_current_span()
            span_context = span.get_span_context()

            if span_context.is_valid:
                log_attrs["trace_id"] = format(span_context.trace_id, "032x")
                log_attrs["span_id"] = format(span_context.span_id, "016x")
                log_attrs["trace_flags"] = span_context.trace_flags

        log_attrs.update(extra_attrs)

        logger.log(level, message, extra=log_attrs)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:  # noqa: PLR0915
        """Execute the agent and stream results back via A2A protocol."""

        start_time = time.time()
        error = self._validate_request(context)
        if error:
            raise A2AServerError("Invalid parameters")

        query = context.get_user_input()
        task_id = context.task_id
        context_id = context.context_id

        tenant_id = self._extract_tenant_id(context)
        set_tenant_id(tenant_id)

        self._log_structured(
            logging.INFO,
            "Agent request started",
            task_id=task_id,
            context_id=context_id,
            tenant_id=tenant_id,
            status="started",
            query_length=len(query),
            query_preview=query[:100] + "..." if len(query) > 100 else query,
        )

        updater = TaskUpdater(event_queue, task_id, context_id)

        with invoke_agent_span(
            provider="anthropic",
            agent_name="CAD Example Agent",
            agent_id="cad-example-agent-v1",
            agent_description="CAD example agent demonstrating debugging and funny answer tools",
            conversation_id=context_id,
            attributes={
                "a2a.task.id": task_id,
                "a2a.context.id": context_id,
                "query.length": len(query),
                "agent.version": "1.0.0",
                "agent.framework": "langchain",
            },
        ) as agent_span:
            agent_span.add_event(
                "agent_execution_started", {"query_preview": query[:100] + "..." if len(query) > 100 else query}
            )

            try:
                # Enqueue a Task object first (required before any status updates in v1.0)
                await event_queue.enqueue_event(
                    Task(
                        id=task_id,
                        context_id=context_id,
                        status=TaskStatus(state=TaskState.submitted),
                    )
                )

                async for item in self.agent.stream(query, context_id):
                    is_task_complete = item["is_task_complete"]
                    require_user_input = item["require_user_input"]
                    content = item.get("content", "")

                    if not is_task_complete and not require_user_input:
                        agent_span.add_event("status_update", {"state": "working", "content_length": len(content)})

                        self._log_structured(
                            logging.DEBUG,
                            "Agent processing task",
                            task_id=task_id,
                            context_id=context_id,
                            tenant_id=tenant_id,
                            status="working",
                            content_length=len(content),
                        )

                        await updater.start_work(updater.new_agent_message([Part(text=content)]))
                    elif require_user_input:
                        agent_span.add_event("input_required", {"reason": "agent_needs_clarification"})

                        self._log_structured(
                            logging.INFO,
                            "Agent requires user input",
                            task_id=task_id,
                            context_id=context_id,
                            tenant_id=tenant_id,
                            status="input_required",
                            reason="agent_needs_clarification",
                        )

                        await updater.requires_input(updater.new_agent_message([Part(text=content)]))
                        break
                    else:
                        agent_span.add_event("agent_completed", {"response_length": len(content), "status": "success"})
                        add_span_attribute("response.character_count", len(content))
                        add_span_attribute("response.word_count", len(content.split()))

                        await updater.add_artifact([Part(text=content)], name="agent_result")
                        await updater.complete()

                        duration_ms = (time.time() - start_time) * 1000
                        # Safely extract token count - handle NonRecordingSpan case
                        total_tokens = 0
                        if agent_span.is_recording():
                            total_tokens = agent_span.attributes.get("gen_ai.usage.total_tokens", 0)

                        record_request_metric(
                            module=Module.AICORE, source=None, operation="invoke_agent", deprecated=False
                        )

                        self._log_structured(
                            logging.INFO,
                            "Agent request completed successfully",
                            task_id=task_id,
                            context_id=context_id,
                            tenant_id=tenant_id,
                            status="success",
                            duration_ms=duration_ms,
                            response_character_count=len(content),
                            response_word_count=len(content.split()),
                            total_tokens=total_tokens,
                        )

                        break

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                # Safely extract token count - handle NonRecordingSpan case
                total_tokens = 0
                if agent_span.is_recording():
                    total_tokens = agent_span.attributes.get("gen_ai.usage.total_tokens", 0)

                record_error_metric(module=Module.AICORE, source=None, operation="invoke_agent", deprecated=False)

                agent_span.add_event("agent_error", {"error.type": type(e).__name__, "error.message": str(e)})
                add_span_attribute("error", True)

                self._log_structured(
                    logging.ERROR,
                    "Agent request failed with exception",
                    task_id=task_id,
                    context_id=context_id,
                    tenant_id=tenant_id,
                    status="failed",
                    duration_ms=duration_ms,
                    total_tokens=total_tokens,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exception=True,
                )

                logger.exception(
                    "Agent request failed with exception",
                    extra={
                        "a2a.task.id": task_id,
                        "a2a.context.id": context_id,
                        "tenant.id": tenant_id,
                        "event.status": "failed",
                    },
                )

                raise A2AServerError("Internal error") from e

    def _validate_request(self, context: RequestContext) -> bool:
        """Validate request. Return False for no error."""
        return False

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel a running task."""
        raise MethodNotImplementedError()