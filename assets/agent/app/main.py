# ruff: noqa: E402
# CRITICAL: Initialize telemetry BEFORE importing AI frameworks (LangChain, LiteLLM, etc.)
import logging
import os
import json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from sap_cloud_sdk.agentgateway._customer import load_customer_credentials, detect_customer_agent_credentials

PROTOCOL = ""

_script_dir = os.path.dirname(os.path.abspath(__file__))
_credentials_dir = os.getenv('SERVICE_BINDING_ROOT', os.path.normpath(os.path.join(_script_dir, "../../../../../credentials")))
_aicore_credentials_file = os.path.join(_credentials_dir, "aicore/credentials.json")

if os.path.exists(_aicore_credentials_file):
    # running locally - always load credentials from file to ensure AICORE env vars are set,
    # regardless of whether SERVICE_BINDING_ROOT is already set in the environment
    if not os.environ.get("SERVICE_BINDING_ROOT"):
        os.environ["SERVICE_BINDING_ROOT"] = _credentials_dir
    PROTOCOL = "http://"
    with open(_aicore_credentials_file) as f:
        content = f.read()
        logger.info(f"Using AI Core credentials from file with content: {content}")
        creds = json.loads(content)
        _AICORE_ENV_KEYS = {"AICORE_CLIENT_ID", "AICORE_CLIENT_SECRET", "AICORE_AUTH_URL", "AICORE_BASE_URL", "AICORE_RESOURCE_GROUP"}
        for key, value in creds.items():
            if key in _AICORE_ENV_KEYS and isinstance(value, str):
                os.environ[key] = value

from sap_cloud_sdk.aicore import set_aicore_config
from sap_cloud_sdk.core.telemetry import auto_instrument

set_aicore_config()
auto_instrument()  # Must be called before importing LangChain/LiteLLM

# Now safe to import AI frameworks and other dependencies

import os
from contextlib import asynccontextmanager

import click
import uvicorn
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from agent_executor import AgentExecutor
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from tools.funny_answers import set_current_jwt_token

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("AGENT_HTTP_PORT") or os.environ.get("PORT", "5000"))
AGENT_PUBLIC_URL = os.environ.get("AGENT_PUBLIC_URL", f"http://{HOST}:{PORT}")

# Agent metadata constants
AGENT_NAME = "CAD Test Agent"
AGENT_VERSION = "1.0.0"

@asynccontextmanager
async def lifespan(app):
    """Lifespan handler for the application."""
    logger.info("CAD Example Agent starting up")
    yield
    logger.info("CAD Example Agent shutting down")


async def health_ready(request):
    """Health readiness endpoint for Kubernetes readiness probes."""
    return JSONResponse({"status": "ready", "service": AGENT_NAME, "version": AGENT_VERSION})


async def health_live(request):
    """Health liveness endpoint for Kubernetes liveness probes."""
    return JSONResponse({"status": "alive", "service": AGENT_NAME, "version": AGENT_VERSION})



@click.command()
@click.option("--host", default=HOST)
@click.option("--port", default=PORT)
def main(host: str, port: int):
    # Agent capabilities
    capabilities = AgentCapabilities(
        streaming=True,
        push_notifications=False,
    )

    solution_id = os.environ.get("SOLUTION_ID", "unknown")
    asset_name = os.environ.get("ASSET_NAME", "unknown")
    client_id = f"{solution_id}-{asset_name}"

    # Funny / easter-egg skill — answers two specific gag questions
    # deterministically (no LLM round-trip):
    #   * "who is the best developer?" → returns the logged-in user from the JWT token
    #   * "when will CAD be delivered?" → "just in time"
    funny_skill = AgentSkill(
        id=f"cad-team-jokes-{client_id}",
        name=f"CAD Team Jokes {client_id}",
        description=(
            f"{client_id} Answers two team in-jokes deterministically (without using the LLM). "
            f"{client_id} 'Who is the best developer?' returns the logged-in user from the OAuth2 JWT token. "
            f"{client_id} 'When will CAD be delivered?' returns 'just in time'."
        ),
        tags=["fun", "easter-egg", "team", "in-joke"],
        examples=[
            f"{client_id} Who is the best developer?",
            f"{client_id} Who is the best developer in the world?",
            f"{client_id} When will CAD be delivered?",
            f"{client_id} When does CAD ship?",
        ],
    )

    # CAD Debugging - MCP Tools Introspection
    # Lists available MCP tools from Agent Gateway (Custom Artefact Deployer debugging feature)
    mcp_tools_skill = AgentSkill(
        id=f"cad-mcp-tools-{client_id}",
        name=f"CAD MCP Tools Introspection {client_id}",
        description=(
            f"{client_id} Custom Artefact Deployer (CAD) debugging functionality. "
            f"{client_id} Lists available MCP tools from the Agent Gateway for introspection and debugging purposes. "
            f"{client_id} This is a CAD-specific debugging feature to verify MCP tool availability."
        ),
        tags=["cad", "debugging", "tools", "mcp", "introspection", "developer", f"{client_id}"],
        examples=[
            f"{client_id} Custom Artefact Deployer list available tools",
            f"{client_id} CAD what tools are available?",
            f"{client_id} Custom Artefact Deployer show me the MCP tools",
            f"{client_id} CAD which capabilities do you have?",
        ],
    )

    # CAD Debugging - S4/HANA Product Query
    # Queries S4/HANA products via MCP (Custom Artefact Deployer debugging feature)
    s4_products_skill = AgentSkill(
        id=f"cad-s4-products-{client_id}",
        name=f"CAD S4 Product Query {client_id}",
        description=(
            f"{client_id} Custom Artefact Deployer (CAD) debugging functionality for S4/HANA integration. "
            f"{client_id} Queries product data from S4/HANA via MCP tool sap.s4:apiResource:API_PRODUCT_0002_MCP:v1. "
            f"{client_id} Uses the list_ProductPlantCosting_for_sap_self tool to retrieve product costing information. "
            f"{client_id} This is a CAD-specific debugging feature to verify S4 connectivity and data retrieval."
        ),
        tags=["cad", "debugging", "s4", "hana", "products", "mcp", f"{client_id}"],
        examples=[
            f"{client_id} Custom Artefact Deployer list products from s4",
            f"{client_id} CAD show products from s4",
            f"{client_id} Custom Artefact Deployer get products from s4",
            f"{client_id} CAD fetch s4 products",
        ],
    )

    # CAD Debugging - Ariba Sourcing Events Query
    # Queries Ariba sourcing events via MCP (Custom Artefact Deployer debugging feature)
    ariba_events_skill = AgentSkill(
        id=f"cad-ariba-events-{client_id}",
        name=f"CAD Ariba Events Query {client_id}",
        description=(
            f"{client_id} Custom Artefact Deployer (CAD) debugging functionality for Ariba integration. "
            f"{client_id} Queries sourcing events from Ariba via MCP tool sap.aribas4:apiResource:sourcing_event_MCP:v1. "
            f"{client_id} Uses the getEventsIds tool to retrieve sourcing event information. "
            f"{client_id} This is a CAD-specific debugging feature to verify Ariba connectivity and data retrieval."
        ),
        tags=["cad", "debugging", "ariba", "sourcing", "events", "mcp", f"{client_id}"],
        examples=[
            f"{client_id} Custom Artefact Deployer list ariba events",
            f"{client_id} CAD show ariba sourcing events",
            f"{client_id} Custom Artefact Deployer get events from ariba",
            f"{client_id} CAD fetch sourcing events",
        ],
    )

    # Agent card (metadata)
    agent_card = AgentCard(
        name=f"{AGENT_NAME} {client_id}",
        description=(
            f"{client_id} A Custom Artefact Deployer (CAD) example agent for debugging and testing. "
            f"{client_id} Demonstrates MCP tool introspection, S4/HANA integration, and Ariba integration capabilities. "
            f"{client_id} Built with Application Foundation."
        ),
        url=AGENT_PUBLIC_URL,
        version=AGENT_VERSION,
        protocol_version="0.3.0",
        preferred_transport="JSONRPC",
        default_input_modes=["text", "text/plain"],
        default_output_modes=["text", "text/plain"],
        capabilities=capabilities,
        skills=[funny_skill, mcp_tools_skill, s4_products_skill, ariba_events_skill],
    )

    # Task store and request handler
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=AgentExecutor(),
        task_store=task_store,
    )

    # Build Starlette app via A2AStarletteApplication, then add health routes
    a2a_app = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
    app = a2a_app.build(lifespan=lifespan)

    # Add middleware to capture JWT token for request context
    class JWTContextMiddleware(BaseHTTPMiddleware):
        """Middleware that extracts JWT token from Authorization header and sets it in context."""

        async def dispatch(self, request, call_next):
            # Extract JWT token from Authorization header
            auth_header = request.headers.get("authorization", "")
            token = None
            if auth_header.lower().startswith("bearer "):
                token = auth_header[7:]  # Remove "Bearer " prefix

            # Set the token in the context variable
            set_current_jwt_token(token)

            try:
                response = await call_next(request)
                return response
            finally:
                # Clear the token after the request
                set_current_jwt_token(None)

    app.add_middleware(JWTContextMiddleware)

    app.routes.extend([
        Route("/health/ready", health_ready),
        Route("/health/live", health_live),
    ])

    logger.info(f"Starting CAD Example Agent at http://{host}:{port}")
    logger.info("Health endpoints: /health/ready (readiness), /health/live (liveness)")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()