import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from core.config import settings

logger = logging.getLogger(__name__)

# Default timeout for API requests (connect timeout, read timeout) in seconds
DEFAULT_TIMEOUT = (5, 30)

# Cache for the loaded private key to avoid repeated file I/O
_private_key_cache: Any = None


def _load_private_key():
    """Load and cache the RSA private key from the configured path."""
    global _private_key_cache

    if _private_key_cache is not None:
        return _private_key_cache

    if not settings.KALSHI_PRIVATE_KEY_PATH:
        logger.warning("No KALSHI_PRIVATE_KEY_PATH configured; requests will not be signed")
        return None

    key_path = Path(settings.KALSHI_PRIVATE_KEY_PATH)
    if not key_path.exists():
        logger.warning("Private key file not found at %s; requests will not be signed", key_path)
        return None

    try:
        with open(key_path, "rb") as key_file:
            _private_key_cache = serialization.load_pem_private_key(
                key_file.read(),
                password=None,  # Assumes unencrypted private key
            )
        logger.info("Successfully loaded private key from %s", key_path)
        return _private_key_cache
    except Exception as exc:
        logger.error("Failed to load private key from %s: %s", key_path, exc)
        return None


def _create_signature(method: str, path: str, timestamp_ms: int, body: str = "") -> Optional[str]:
    """
    Create RSA signature for Kalshi API authentication.

    The signature message format is: {timestamp_ms}{method}{path}{body}
    where body is the JSON string if present, otherwise empty string.
    """
    private_key = _load_private_key()
    if private_key is None:
        return None

    # Construct the message to sign
    message = f"{timestamp_ms}{method}{path}{body}"
    message_bytes = message.encode("utf-8")

    try:
        # Sign with RSA-PSS padding and SHA-256 hash
        signature = private_key.sign(
            message_bytes,
            padding.PKCS1v15(),  # Kalshi uses PKCS1v15, not PSS
            hashes.SHA256(),
        )
        # Return base64-encoded signature
        return base64.b64encode(signature).decode("utf-8")
    except Exception as exc:
        logger.error("Failed to create signature: %s", exc)
        return None


class KalshiRateLimitError(Exception):
    """Raised when Kalshi API returns 429 (rate limit exceeded)."""
    pass


class KalshiServerError(Exception):
    """Raised when Kalshi API returns 5xx (server error)."""
    pass


def _should_retry(response: requests.Response) -> None:
    """
    Check if the response indicates a retryable error.
    Raises an exception if retry is needed, otherwise does nothing.
    """
    if response.status_code == 429:
        raise KalshiRateLimitError(f"Rate limit exceeded: {response.text}")
    elif 500 <= response.status_code < 600:
        raise KalshiServerError(f"Server error {response.status_code}: {response.text}")
    # Otherwise, no retry needed
    response.raise_for_status()  # Raise for other 4xx errors


@retry(
    retry=retry_if_exception_type((KalshiRateLimitError, KalshiServerError, requests.Timeout)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def kalshi_request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    timeout: tuple[float, float] | float | None = None,
    **kwargs
) -> requests.Response:
    """
    Wrapper for Kalshi requests with RSA signature authentication and automatic retries.

    Automatically:
    - Cleans None values from params and json payloads
    - Adds authentication headers (signature and timestamp)
    - Constructs the full API URL
    - Retries on 429 (rate limit), 5xx (server errors), and timeouts with exponential backoff

    Args:
        method: HTTP method (GET, POST, etc.)
        path: API path (e.g., '/markets')
        params: Query parameters
        json: JSON body payload
        timeout: Request timeout in seconds. Can be a float (total timeout) or
                 tuple (connect_timeout, read_timeout). Defaults to (5, 30).
        **kwargs: Additional arguments passed to requests.request()
    """
    # Apply default timeout if not specified
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    # Clean None values from params
    if params:
        params = {k: v for k, v in params.items() if v is not None}

    # Clean None values from json payload
    if json:
        json = {k: v for k, v in json.items() if v is not None}

    # Construct URL
    base_url = settings.KALSHI_BASE_URL.rstrip("/")
    if "/trade-api" not in base_url:
        base_url = f"{base_url}/trade-api/v2"
    url_path = path if path.startswith("/") else f"/{path}"
    url = f"{base_url}{url_path}"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Add authentication if key is available
    timestamp_ms = int(time.time() * 1000)
    body_str = json.dumps(json) if json else ""
    signature = _create_signature(method.upper(), url_path, timestamp_ms, body_str)

    if signature and settings.KALSHI_KEY_ID:
        headers["KALSHI-ACCESS-KEY"] = settings.KALSHI_KEY_ID
        headers["KALSHI-ACCESS-SIGNATURE"] = signature
        headers["KALSHI-ACCESS-TIMESTAMP"] = str(timestamp_ms)

    # Make the request with timeout
    response = requests.request(
        method,
        url,
        headers=headers,
        params=params,
        json=json,
        timeout=timeout,
        **kwargs
    )

    # Check if we should retry
    _should_retry(response)

    return response
