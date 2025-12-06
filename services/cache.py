"""
Simple in-memory cache with TTL support for API responses.
Provides caching for Kalshi API calls to reduce redundant requests.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with expiration."""

    value: T
    expires_at: float

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


@dataclass
class TTLCache(Generic[T]):
    """
    Thread-safe TTL cache for API responses.

    Features:
    - Configurable TTL (time-to-live) per cache instance
    - Automatic cleanup of expired entries
    - Thread-safe operations
    - Max size limit to prevent unbounded growth
    """

    default_ttl: float = 300.0  # 5 minutes default
    max_size: int = 1000
    _entries: Dict[str, CacheEntry[T]] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _last_cleanup: float = field(default_factory=time.time)
    _cleanup_interval: float = 60.0  # Run cleanup every minute

    def get(self, key: str) -> Optional[T]:
        """Get a value from cache if it exists and is not expired."""
        with self._lock:
            self._maybe_cleanup()

            entry = self._entries.get(key)
            if entry is None:
                return None

            if entry.is_expired():
                del self._entries[key]
                return None

            return entry.value

    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Set a value in cache with optional custom TTL."""
        with self._lock:
            self._maybe_cleanup()

            # Enforce max size by removing oldest entries
            if len(self._entries) >= self.max_size:
                self._evict_oldest()

            expires_at = time.time() + (ttl if ttl is not None else self.default_ttl)
            self._entries[key] = CacheEntry(value=value, expires_at=expires_at)

    def delete(self, key: str) -> bool:
        """Delete a key from cache. Returns True if key existed."""
        with self._lock:
            if key in self._entries:
                del self._entries[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._entries.clear()

    def size(self) -> int:
        """Return number of entries in cache (including expired)."""
        with self._lock:
            return len(self._entries)

    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed since last cleanup."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = now

    def _cleanup_expired(self) -> None:
        """Remove all expired entries."""
        now = time.time()
        expired_keys = [
            key for key, entry in self._entries.items() if entry.expires_at < now
        ]
        for key in expired_keys:
            del self._entries[key]
        if expired_keys:
            logger.debug("Cleaned up %d expired cache entries", len(expired_keys))

    def _evict_oldest(self) -> None:
        """Remove the oldest entry to make room for new ones."""
        if not self._entries:
            return
        # Find entry with earliest expiration
        oldest_key = min(self._entries.keys(), key=lambda k: self._entries[k].expires_at)
        del self._entries[oldest_key]


def cached(
    cache: TTLCache,
    key_fn: Callable[..., str],
    ttl: Optional[float] = None
) -> Callable:
    """
    Decorator for caching function results.

    Args:
        cache: TTLCache instance to use
        key_fn: Function that takes the same args as the decorated function
                and returns a cache key string
        ttl: Optional TTL override for this specific cache

    Example:
        @cached(my_cache, key_fn=lambda ticker: f"market:{ticker}")
        def get_market(ticker: str) -> dict:
            return api.fetch(ticker)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            key = key_fn(*args, **kwargs)

            # Try cache first
            cached_value = cache.get(key)
            if cached_value is not None:
                logger.debug("Cache hit for key: %s", key)
                return cached_value

            # Cache miss - call function
            logger.debug("Cache miss for key: %s", key)
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(key, result, ttl)

            return result
        return wrapper
    return decorator


# Global cache instances for different data types
_market_cache: Optional[TTLCache] = None
_events_cache: Optional[TTLCache] = None


def get_market_cache() -> TTLCache:
    """Get the global market cache instance."""
    global _market_cache
    if _market_cache is None:
        _market_cache = TTLCache(
            default_ttl=60.0,  # 1 minute for market data (prices change often)
            max_size=500,
        )
    return _market_cache


def get_events_cache() -> TTLCache:
    """Get the global events list cache instance."""
    global _events_cache
    if _events_cache is None:
        _events_cache = TTLCache(
            default_ttl=300.0,  # 5 minutes for events list (changes less often)
            max_size=100,
        )
    return _events_cache


def clear_all_caches() -> None:
    """Clear all cache instances. Useful for testing."""
    global _market_cache, _events_cache
    if _market_cache:
        _market_cache.clear()
    if _events_cache:
        _events_cache.clear()
