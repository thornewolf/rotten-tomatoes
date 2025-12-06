import logging

import requests

from core.config import settings

logger = logging.getLogger(__name__)


def is_auth_configured() -> bool:
    return all(
        [
            settings.KALSHI_API_KEY,
            settings.KALSHI_PRIVATE_KEY_PATH,
            settings.KALSHI_KEY_ID,
        ]
    )


def kalshi_request(method: str, path: str, **kwargs) -> requests.Response:
    """
    Wrapper for Kalshi requests.
    In a real implementation, this handles RSA signing headers.
    """
    base_url = "https://trading-api.kalshi.com/trade-api/v2"
    url = f"{base_url}{path}"
    headers = {"Content-Type": "application/json"}
    return requests.request(method, url, headers=headers, **kwargs)
