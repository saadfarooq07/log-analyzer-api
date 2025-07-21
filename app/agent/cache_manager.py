"""Caching layer with performance metrics for the log analyzer.

This module provides intelligent caching to improve performance and
reduce redundant API calls.
"""

import time
import hashlib
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
import asyncio
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class CacheEntry:
    """Represents a single cache entry."""
    
    def __init__(self, key: str, value: Any, ttl: int = 300):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 1
        self.size_bytes = len(json.dumps(value))
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.created_at > self.ttl
    
    def access(self) -> Any:
        """Access the cache entry and update metadata."""
        self.last_accessed = time.time()
        self.access_count += 1
        return self.value
    
    def get_age(self) -> float:
        """Get age of the cache entry in seconds."""
        return time.time() - self.created_at


class LRUCache:
    """Least Recently Used (LRU) cache implementation."""
    
    def __init__(self, max_size: int = 100, max_memory_mb: float = 100.0):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.total_memory_bytes = 0
        
        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_requests = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        self.total_requests += 1
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove(key)
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return entry.access()
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any, ttl: int = 300) -> None:
        """Put value in cache."""
        # Remove if already exists
        if key in self.cache:
            self._remove(key)
        
        # Create new entry
        entry = CacheEntry(key, value, ttl)
        
        # Check memory limit
        while (self.total_memory_bytes + entry.size_bytes > self.max_memory_bytes 
               and len(self.cache) > 0):
            self._evict_lru()
        
        # Check size limit
        while len(self.cache) >= self.max_size:
            self._evict_lru()
        
        # Add to cache
        self.cache[key] = entry
        self.total_memory_bytes += entry.size_bytes
    
    def _remove(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.total_memory_bytes -= entry.size_bytes
            del self.cache[key]
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self.cache:
            key, entry = self.cache.popitem(last=False)
            self.total_memory_bytes -= entry.size_bytes
            self.evictions += 1
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.total_memory_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = (self.hits / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "memory_mb": self.total_memory_bytes / (1024 * 1024),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "total_requests": self.total_requests
        }


class AnalysisCache:
    """Specialized cache for log analysis results."""
    
    def __init__(self):
        # Different caches for different types of data
        self.analysis_cache = LRUCache(max_size=50, max_memory_mb=200)
        self.pattern_cache = LRUCache(max_size=200, max_memory_mb=50)
        self.doc_cache = LRUCache(max_size=100, max_memory_mb=100)
        
        # Performance tracking
        self.operation_times: Dict[str, List[float]] = {
            "analysis": [],
            "pattern_extraction": [],
            "doc_search": []
        }
    
    def get_cache_key(self, log_content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key from log content and metadata."""
        # Create a deterministic key
        key_parts = [log_content[:1000]]  # First 1000 chars
        
        if metadata:
            # Add relevant metadata to key
            for k in sorted(["analysis_type", "application_name", "environment_details"]):
                if k in metadata:
                    key_parts.append(f"{k}:{metadata[k]}")
        
        key_string = "|".join(str(p) for p in key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get_analysis(self, log_content: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get cached analysis result."""
        key = self.get_cache_key(log_content, metadata)
        return self.analysis_cache.get(key)
    
    def cache_analysis(self, log_content: str, result: Dict[str, Any], 
                      metadata: Optional[Dict[str, Any]] = None, ttl: int = 300) -> None:
        """Cache analysis result."""
        key = self.get_cache_key(log_content, metadata)
        self.analysis_cache.put(key, result, ttl)
    
    def get_pattern(self, pattern_type: str, log_content: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached pattern extraction result."""
        key = f"{pattern_type}:{hashlib.md5(log_content.encode()).hexdigest()[:16]}"
        return self.pattern_cache.get(key)
    
    def cache_pattern(self, pattern_type: str, log_content: str, 
                     patterns: List[Dict[str, Any]], ttl: int = 600) -> None:
        """Cache pattern extraction result."""
        key = f"{pattern_type}:{hashlib.md5(log_content.encode()).hexdigest()[:16]}"
        self.pattern_cache.put(key, patterns, ttl)
    
    def get_doc_search(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached documentation search result."""
        key = f"doc:{hashlib.md5(query.encode()).hexdigest()[:16]}"
        return self.doc_cache.get(key)
    
    def cache_doc_search(self, query: str, results: List[Dict[str, Any]], ttl: int = 3600) -> None:
        """Cache documentation search result."""
        key = f"doc:{hashlib.md5(query.encode()).hexdigest()[:16]}"
        self.doc_cache.put(key, results, ttl)
    
    def record_operation_time(self, operation: str, duration: float) -> None:
        """Record operation execution time."""
        if operation in self.operation_times:
            self.operation_times[operation].append(duration)
            # Keep only last 100 times
            if len(self.operation_times[operation]) > 100:
                self.operation_times[operation] = self.operation_times[operation][-100:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "caches": {
                "analysis": self.analysis_cache.get_stats(),
                "pattern": self.pattern_cache.get_stats(),
                "documentation": self.doc_cache.get_stats()
            },
            "operation_times": {}
        }
        
        # Calculate operation time statistics
        for op, times in self.operation_times.items():
            if times:
                stats["operation_times"][op] = {
                    "count": len(times),
                    "avg_ms": sum(times) / len(times) * 1000,
                    "min_ms": min(times) * 1000,
                    "max_ms": max(times) * 1000,
                    "p95_ms": sorted(times)[int(len(times) * 0.95)] * 1000 if len(times) > 1 else times[0] * 1000
                }
        
        # Overall metrics
        total_hits = sum(c["hits"] for c in stats["caches"].values())
        total_requests = sum(c["total_requests"] for c in stats["caches"].values())
        
        stats["overall"] = {
            "total_hits": total_hits,
            "total_requests": total_requests,
            "overall_hit_rate": (total_hits / total_requests * 100) if total_requests > 0 else 0,
            "total_memory_mb": sum(c["memory_mb"] for c in stats["caches"].values())
        }
        
        return stats
    
    def clear_all(self) -> None:
        """Clear all caches."""
        self.analysis_cache.clear()
        self.pattern_cache.clear()
        self.doc_cache.clear()


def cached_operation(cache_type: str = "analysis", ttl: int = 300):
    """Decorator for caching operation results."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            # Get cache manager
            cache_manager = get_cache_manager()
            
            # Generate cache key based on function name and args
            key_parts = [func.__name__] + [str(arg) for arg in args] + [f"{k}:{v}" for k, v in sorted(kwargs.items())]
            cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            start_time = time.time()
            
            if cache_type == "analysis":
                cached = cache_manager.analysis_cache.get(cache_key)
            elif cache_type == "pattern":
                cached = cache_manager.pattern_cache.get(cache_key)
            elif cache_type == "doc":
                cached = cache_manager.doc_cache.get(cache_key)
            else:
                cached = None
            
            if cached is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached
            
            # Execute function
            result = await func(self, *args, **kwargs)
            
            # Cache result
            if cache_type == "analysis":
                cache_manager.analysis_cache.put(cache_key, result, ttl)
            elif cache_type == "pattern":
                cache_manager.pattern_cache.put(cache_key, result, ttl)
            elif cache_type == "doc":
                cache_manager.doc_cache.put(cache_key, result, ttl)
            
            # Record operation time
            duration = time.time() - start_time
            cache_manager.record_operation_time(func.__name__, duration)
            
            return result
        
        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            # Similar logic for sync functions
            cache_manager = get_cache_manager()
            
            key_parts = [func.__name__] + [str(arg) for arg in args] + [f"{k}:{v}" for k, v in sorted(kwargs.items())]
            cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            start_time = time.time()
            
            if cache_type == "analysis":
                cached = cache_manager.analysis_cache.get(cache_key)
            elif cache_type == "pattern":
                cached = cache_manager.pattern_cache.get(cache_key)
            elif cache_type == "doc":
                cached = cache_manager.doc_cache.get(cache_key)
            else:
                cached = None
            
            if cached is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached
            
            result = func(self, *args, **kwargs)
            
            if cache_type == "analysis":
                cache_manager.analysis_cache.put(cache_key, result, ttl)
            elif cache_type == "pattern":
                cache_manager.pattern_cache.put(cache_key, result, ttl)
            elif cache_type == "doc":
                cache_manager.doc_cache.put(cache_key, result, ttl)
            
            duration = time.time() - start_time
            cache_manager.record_operation_time(func.__name__, duration)
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Global cache manager
_cache_manager = AnalysisCache()


def get_cache_manager() -> AnalysisCache:
    """Get the global cache manager."""
    return _cache_manager