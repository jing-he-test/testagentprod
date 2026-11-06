"""
Common utilities shared across agent tools.

Provides:
- JWT token context management (for request-scoped token access)
- JWT payload decoding and user extraction
- Lazy-initialized Agent Gateway client
"""

from __future__ import annotations

import base64
import json
import logging
from contextvars import ContextVar

from sap_cloud_sdk.agentgateway import create_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JWT Token Context Management
# ---------------------------------------------------------------------------

# Context variable to hold the current JWT token for the request
# This should be set by middleware or the request handler
_current_jwt_token: ContextVar[str | None] = ContextVar("current_jwt_token", default=None)


def set_current_jwt_token(token: str | None) -> None:
    """Set the JWT token for the current request context."""
    _current_jwt_token.set(token)


def get_current_jwt_token() -> str | None:
    """Get the JWT token from the current request context."""
    return _current_jwt_token.get()


# ---------------------------------------------------------------------------
# JWT Decoding Utilities
# ---------------------------------------------------------------------------

def decode_jwt_payload(token: str) -> dict:
    """
    Decode the payload from a JWT token without verifying the signature.

    JWT tokens have three parts separated by dots: header.payload.signature
    The payload is base64url encoded JSON.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return {}

        # Decode the payload (second part)
        payload_b64 = parts[1]
        # Add padding if needed (base64url requires padding to multiple of 4)
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding

        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        return json.loads(payload_bytes)
    except Exception as e:
        logger.warning("Failed to decode JWT payload: %s", e)
        return {}


def extract_user_from_jwt(token: str) -> str | None:
    """
    Extract the user identifier from a JWT token.

    Looks for common claims that contain user information:
    - 'email' (most common for IAS tokens)
    - 'preferred_username'
    - 'sub' (subject)
    - 'user_name'
    - 'name'
    """
    payload = decode_jwt_payload(token)
    if not payload:
        return None

    logger.info("Decoded JWT payload: %s", payload)

    # Try common user identifier claims in order of preference
    if 'given_name' in payload and 'family_name' in payload:
        return str(payload['given_name']) + " " + str(payload['family_name'])
    elif 'sub' in payload:
        return str(payload['sub'])
    elif 'email' in payload:
        return str(payload['email'])
    elif 'preferred_username' in payload:
        return str(payload['preferred_username'])
    elif 'user_name' in payload:
        return str(payload['user_name'])
    elif 'name' in payload:
        return str(payload['name'])
    else:
        return "Philipp Herzig"


def get_user_from_context() -> str | None:
    """
    Attempt to extract the user from the current context variable.

    The JWT token should be set via set_current_jwt_token() by middleware
    or the request handler.
    """
    token = get_current_jwt_token()
    if token:
        return extract_user_from_jwt(token)
    return None


# ---------------------------------------------------------------------------
# Agent Gateway Client
# ---------------------------------------------------------------------------

# Lazy-initialized client (created on first use, not at import time)
_agent_gateway_client = None


def get_agent_gateway_client():
    """Get or create the agent gateway client (lazy initialization)."""
    global _agent_gateway_client
    if _agent_gateway_client is None:
        logger.info("Creating agent gateway client...")
        _agent_gateway_client = create_client()
    return _agent_gateway_client