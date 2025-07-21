"""Rate limiting implementation for API protection.

This module provides rate limiting functionality to prevent API quota exhaustion
and ensure fair usage of external services.
"""

import time
import asyncio
from typing import Dict, Any, Optional, Callable, TypeVar
from functools import wraps
from collections import deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


class TokenBucket:
    """Token bucket algorithm for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired, False otherwise
        """
        async with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def acquire_sync(self, tokens: int = 1) -> bool:
        """Synchronous version of acquire."""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on refill rate
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait for tokens to be available."""
        self._refill()
        
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class SlidingWindowRateLimiter:
    """Sliding window rate limiter for more precise control."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        """Initialize sliding window rate limiter.
        
        Args:
            max_requests: Maximum requests in window
            window_seconds: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque = deque()
        self._lock = asyncio.Lock()
        
        # Metrics
        self.total_requests = 0
        self.rejected_requests = 0
    
    async def acquire(self) -> bool:
        """Try to acquire permission for a request."""
        async with self._lock:
            return self._acquire_internal()
    
    def acquire_sync(self) -> bool:
        """Synchronous version of acquire."""
        return self._acquire_internal()
    
    def _acquire_internal(self) -> bool:
        """Internal acquire logic."""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Remove old requests outside window
        while self.requests and self.requests[0] < window_start:
            self.requests.popleft()
        
        self.total_requests += 1
        
        # Check if we can make a request
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        else:
            self.rejected_requests += 1
            return False
    
    def get_wait_time(self) -> float:
        """Get time to wait before next request is allowed."""
        if len(self.requests) < self.max_requests:
            return 0.0
        
        # Wait until oldest request exits window
        oldest = self.requests[0]
        now = time.time()
        wait_time = (oldest + self.window_seconds) - now
        
        return max(0.0, wait_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "current_requests": len(self.requests),
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "total_requests": self.total_requests,
            "rejected_requests": self.rejected_requests,
            "rejection_rate": (self.rejected_requests / self.total_requests * 100) if self.total_requests > 0 else 0.0
        }


class RateLimiter:
    """Composite rate limiter supporting multiple algorithms."""
    
    def __init__(
        self,
        name: str,
        requests_per_minute: Optional[int] = None,
        requests_per_hour: Optional[int] = None,
        burst_capacity: Optional[int] = None,
        tokens_per_second: Optional[float] = None
    ):
        """Initialize rate limiter.
        
        Args:
            name: Name of the rate limiter
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour
            burst_capacity: Max burst capacity for token bucket
            tokens_per_second: Token refill rate
        """
        self.name = name
        self.limiters = []
        
        # Add sliding window limiters
        if requests_per_minute:
            self.limiters.append(
                SlidingWindowRateLimiter(requests_per_minute, 60)
            )
        
        if requests_per_hour:
            self.limiters.append(
                SlidingWindowRateLimiter(requests_per_hour, 3600)
            )
        
        # Add token bucket for burst control
        if burst_capacity and tokens_per_second:
            self.limiters.append(
                TokenBucket(burst_capacity, tokens_per_second)
            )
        elif requests_per_minute:
            # Default token bucket based on per-minute rate
            self.limiters.append(
                TokenBucket(
                    capacity=max(10, requests_per_minute // 6),
                    refill_rate=requests_per_minute / 60.0
                )
            )
    
    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        for limiter in self.limiters:
            if isinstance(limiter, TokenBucket):
                if not await limiter.acquire():
                    wait_time = limiter.get_wait_time()
                    raise RateLimitExceeded(
                        f"Rate limit exceeded for '{self.name}'. "
                        f"Please wait {wait_time:.1f} seconds."
                    )
            elif isinstance(limiter, SlidingWindowRateLimiter):
                if not await limiter.acquire():
                    wait_time = limiter.get_wait_time()
                    raise RateLimitExceeded(
                        f"Rate limit exceeded for '{self.name}'. "
                        f"Please wait {wait_time:.1f} seconds."
                    )
    
    def acquire_sync(self) -> None:
        """Synchronous version of acquire."""
        for limiter in self.limiters:
            if isinstance(limiter, TokenBucket):
                if not limiter.acquire_sync():
                    wait_time = limiter.get_wait_time()
                    raise RateLimitExceeded(
                        f"Rate limit exceeded for '{self.name}'. "
                        f"Please wait {wait_time:.1f} seconds."
                    )
            elif isinstance(limiter, SlidingWindowRateLimiter):
                if not limiter.acquire_sync():
                    wait_time = limiter.get_wait_time()
                    raise RateLimitExceeded(
                        f"Rate limit exceeded for '{self.name}'. "
                        f"Please wait {wait_time:.1f} seconds."
                    )
    
    async def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        max_wait = 0.0
        
        for limiter in self.limiters:
            if isinstance(limiter, (TokenBucket, SlidingWindowRateLimiter)):
                wait_time = limiter.get_wait_time()
                max_wait = max(max_wait, wait_time)
        
        if max_wait > 0:
            logger.info(f"Rate limiter '{self.name}' waiting {max_wait:.1f}s")
            await asyncio.sleep(max_wait)
    
    def decorator(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for applying rate limiting to functions."""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            await self.acquire()
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            self.acquire_sync()
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        stats = {
            "name": self.name,
            "limiters": []
        }
        
        for limiter in self.limiters:
            if isinstance(limiter, SlidingWindowRateLimiter):
                stats["limiters"].append({
                    "type": "sliding_window",
                    **limiter.get_stats()
                })
            elif isinstance(limiter, TokenBucket):
                stats["limiters"].append({
                    "type": "token_bucket",
                    "tokens": limiter.tokens,
                    "capacity": limiter.capacity,
                    "refill_rate": limiter.refill_rate
                })
        
        return stats


class APIRateLimiters:
    """Pre-configured rate limiters for common APIs."""
    
    @staticmethod
    def gemini() -> RateLimiter:
        """Rate limiter for Gemini API (60 requests/minute)."""
        return RateLimiter(
            name="gemini",
            requests_per_minute=60,
            burst_capacity=10,
            tokens_per_second=1.0
        )
    
    @staticmethod
    def groq() -> RateLimiter:
        """Rate limiter for Groq API (30 requests/minute)."""
        return RateLimiter(
            name="groq",
            requests_per_minute=30,
            burst_capacity=5,
            tokens_per_second=0.5
        )
    
    @staticmethod
    def tavily() -> RateLimiter:
        """Rate limiter for Tavily API (100 requests/minute)."""
        return RateLimiter(
            name="tavily",
            requests_per_minute=100,
            burst_capacity=20,
            tokens_per_second=1.67
        )


# Global rate limiter instances
_rate_limiters: Dict[str, RateLimiter] = {}


def get_rate_limiter(name: str, **kwargs) -> RateLimiter:
    """Get or create a rate limiter."""
    if name not in _rate_limiters:
        # Check for pre-configured limiters
        if name == "gemini":
            _rate_limiters[name] = APIRateLimiters.gemini()
        elif name == "groq":
            _rate_limiters[name] = APIRateLimiters.groq()
        elif name == "tavily":
            _rate_limiters[name] = APIRateLimiters.tavily()
        else:
            # Create custom limiter
            _rate_limiters[name] = RateLimiter(name=name, **kwargs)
    
    return _rate_limiters[name]