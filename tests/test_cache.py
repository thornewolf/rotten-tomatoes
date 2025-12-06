"""
Unit tests for the caching module.
"""

import pytest
import time
from unittest.mock import patch

from services.cache import (
    TTLCache,
    CacheEntry,
    cached,
    get_market_cache,
    get_events_cache,
    clear_all_caches,
)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_is_expired_false_when_valid(self):
        """Entry should not be expired before expires_at."""
        entry = CacheEntry(value="test", expires_at=time.time() + 100)
        assert entry.is_expired() is False

    def test_is_expired_true_when_expired(self):
        """Entry should be expired after expires_at."""
        entry = CacheEntry(value="test", expires_at=time.time() - 1)
        assert entry.is_expired() is True


class TestTTLCache:
    """Tests for TTLCache functionality."""

    def test_set_and_get(self):
        """Should store and retrieve values."""
        cache = TTLCache[str]()
        cache.set("key1", "value1")

        assert cache.get("key1") == "value1"

    def test_get_missing_key_returns_none(self):
        """Should return None for missing keys."""
        cache = TTLCache[str]()
        assert cache.get("nonexistent") is None

    def test_get_expired_entry_returns_none(self):
        """Should return None for expired entries."""
        cache = TTLCache[str](default_ttl=0.01)
        cache.set("key1", "value1")

        time.sleep(0.02)  # Wait for expiration

        assert cache.get("key1") is None

    def test_custom_ttl(self):
        """Should respect custom TTL per entry."""
        cache = TTLCache[str](default_ttl=100)
        cache.set("short", "value", ttl=0.01)
        cache.set("long", "value", ttl=100)

        time.sleep(0.02)

        assert cache.get("short") is None
        assert cache.get("long") == "value"

    def test_delete(self):
        """Should delete entries."""
        cache = TTLCache[str]()
        cache.set("key1", "value1")

        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("key1") is False  # Already deleted

    def test_clear(self):
        """Should clear all entries."""
        cache = TTLCache[str]()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.size() == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_size(self):
        """Should report correct size."""
        cache = TTLCache[str]()
        assert cache.size() == 0

        cache.set("key1", "value1")
        assert cache.size() == 1

        cache.set("key2", "value2")
        assert cache.size() == 2

    def test_max_size_eviction(self):
        """Should evict oldest entries when max_size reached."""
        cache = TTLCache[str](max_size=3, default_ttl=100)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # This should trigger eviction
        cache.set("key4", "value4")

        assert cache.size() <= 3
        # key4 should exist
        assert cache.get("key4") == "value4"

    def test_thread_safety(self):
        """Cache should be thread-safe."""
        import threading

        cache = TTLCache[int]()
        errors = []

        def writer():
            try:
                for i in range(100):
                    cache.set(f"key{i}", i)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    cache.get(f"key{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestCachedDecorator:
    """Tests for the @cached decorator."""

    def test_cached_function_result(self):
        """Should cache function results."""
        cache = TTLCache[int]()
        call_count = 0

        @cached(cache, key_fn=lambda x: f"key:{x}")
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call - should execute function
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call - should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Still 1, used cache

    def test_cached_different_keys(self):
        """Should cache different keys separately."""
        cache = TTLCache[int]()
        call_count = 0

        @cached(cache, key_fn=lambda x: f"key:{x}")
        def add_one(x):
            nonlocal call_count
            call_count += 1
            return x + 1

        assert add_one(1) == 2
        assert add_one(2) == 3
        assert call_count == 2

        # Cache hits
        assert add_one(1) == 2
        assert add_one(2) == 3
        assert call_count == 2


class TestGlobalCaches:
    """Tests for global cache instances."""

    def test_get_market_cache_singleton(self):
        """Should return same instance."""
        cache1 = get_market_cache()
        cache2 = get_market_cache()
        assert cache1 is cache2

    def test_get_events_cache_singleton(self):
        """Should return same instance."""
        cache1 = get_events_cache()
        cache2 = get_events_cache()
        assert cache1 is cache2

    def test_clear_all_caches(self):
        """Should clear all global caches."""
        market_cache = get_market_cache()
        events_cache = get_events_cache()

        market_cache.set("test", "value")
        events_cache.set("test", "value")

        clear_all_caches()

        assert market_cache.get("test") is None
        assert events_cache.get("test") is None
