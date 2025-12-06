import logging

import requests

from core.config import settings

logger = logging.getLogger(__name__)


def kalshi_request(method: str, path: str, **kwargs) -> requests.Response:
    """
    Wrapper for Kalshi requests.
    In a real implementation, this handles RSA signing headers.
    """
    base_url = settings.KALSHI_BASE_URL.rstrip("/")
    if "/trade-api" not in base_url:
        base_url = f"{base_url}/trade-api/v2"
    url_path = path if path.startswith("/") else f"/{path}"
    url = f"{base_url}{url_path}"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    return requests.request(method, url, headers=headers, **kwargs)
