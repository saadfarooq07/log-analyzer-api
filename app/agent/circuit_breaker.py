"""Circuit breaker implementation for fault tolerance.

This module provides circuit breaker functionality to prevent cascading failures
and protect external services from being overwhelmed during outages.
"""

import time
import asyncio
from typing import Dict, Any, Optional, Callable, TypeVar, Union
from enum import Enum
from functools import wraps
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        success_threshold: int = 2
    ):
        """Initialize circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
            success_threshold: Successes needed to close circuit from half-open
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        
        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_state_change: float = time.time()
        
        # Metrics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._circuit_open_count = 0
        
    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for recovery timeout."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and \
               time.time() - self._last_failure_time >= self.recovery_timeout:
                logger.info(f"Circuit breaker '{self.name}' moving to HALF_OPEN state")
                self._state = CircuitState.HALF_OPEN
                self._last_state_change = time.time()
        return self._state
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        return asyncio.run(self.async_call(func, *args, **kwargs)) if asyncio.iscoroutinefunction(func) else self._sync_call(func, *args, **kwargs)
    
    def _sync_call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Synchronous call execution."""
        self._total_calls += 1
        
        if self.state == CircuitState.OPEN:
            self._on_circuit_open()
            raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    async def async_call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Asynchronous call execution."""
        self._total_calls += 1
        
        if self.state == CircuitState.OPEN:
            self._on_circuit_open()
            raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call."""
        self._total_successes += 1
        
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            logger.debug(f"Circuit breaker '{self.name}' success in HALF_OPEN: {self._success_count}/{self.success_threshold}")
            
            if self._success_count >= self.success_threshold:
                logger.info(f"Circuit breaker '{self.name}' closing after recovery")
                self._state = CircuitState.CLOSED
                self._last_state_change = time.time()
                self._failure_count = 0
                self._success_count = 0
        else:
            # Reset failure count on success in CLOSED state
            self._failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self._total_failures += 1
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit breaker '{self.name}' failure in HALF_OPEN, reopening")
            self._state = CircuitState.OPEN
            self._last_state_change = time.time()
            self._circuit_open_count += 1
            self._success_count = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            logger.debug(f"Circuit breaker '{self.name}' failure: {self._failure_count}/{self.failure_threshold}")
            
            if self._failure_count >= self.failure_threshold:
                logger.error(f"Circuit breaker '{self.name}' opening after {self._failure_count} failures")
                self._state = CircuitState.OPEN
                self._last_state_change = time.time()
                self._circuit_open_count += 1
    
    def _on_circuit_open(self):
        """Handle circuit open event."""
        logger.warning(f"Circuit breaker '{self.name}' rejecting call (circuit OPEN)")
    
    def reset(self):
        """Manually reset the circuit breaker."""
        logger.info(f"Circuit breaker '{self.name}' manually reset")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._last_state_change = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "circuit_open_count": self._circuit_open_count,
            "last_failure_time": datetime.fromtimestamp(self._last_failure_time).isoformat() if self._last_failure_time else None,
            "last_state_change": datetime.fromtimestamp(self._last_state_change).isoformat(),
            "uptime_percentage": (self._total_successes / self._total_calls * 100) if self._total_calls > 0 else 100.0
        }
    
    def decorator(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for applying circuit breaker to functions."""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self.async_call(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


class CircuitBreakerManager:
    """Manages multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
    
    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        success_threshold: int = 2
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception,
                success_threshold=success_threshold
            )
        return self._breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all circuit breakers."""
        total_calls = sum(b._total_calls for b in self._breakers.values())
        total_failures = sum(b._total_failures for b in self._breakers.values())
        open_circuits = sum(1 for b in self._breakers.values() if b.state == CircuitState.OPEN)
        
        return {
            "total_breakers": len(self._breakers),
            "open_circuits": open_circuits,
            "total_calls": total_calls,
            "total_failures": total_failures,
            "overall_success_rate": ((total_calls - total_failures) / total_calls * 100) if total_calls > 0 else 100.0,
            "breaker_states": {name: breaker.state.value for name, breaker in self._breakers.items()}
        }


# Global circuit breaker manager
_manager = CircuitBreakerManager()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception,
    success_threshold: int = 2
) -> CircuitBreaker:
    """Get or create a circuit breaker with the given configuration."""
    return _manager.get_or_create(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception,
        success_threshold=success_threshold
    )


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager."""
    return _manager