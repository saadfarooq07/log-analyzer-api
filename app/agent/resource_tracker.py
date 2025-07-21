"""Resource tracking and management for the log analyzer.

This module provides comprehensive resource monitoring to prevent
resource exhaustion and optimize performance.
"""

import psutil
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    open_files: int
    threads: int
    
    @classmethod
    def capture(cls) -> "ResourceSnapshot":
        """Capture current resource usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        io_counters = process.io_counters() if hasattr(process, "io_counters") else None
        net_io = psutil.net_io_counters()
        
        return cls(
            timestamp=time.time(),
            cpu_percent=process.cpu_percent(interval=0.1),
            memory_percent=process.memory_percent(),
            memory_mb=memory_info.rss / (1024 * 1024),
            disk_io_read_mb=io_counters.read_bytes / (1024 * 1024) if io_counters else 0,
            disk_io_write_mb=io_counters.write_bytes / (1024 * 1024) if io_counters else 0,
            network_sent_mb=net_io.bytes_sent / (1024 * 1024),
            network_recv_mb=net_io.bytes_recv / (1024 * 1024),
            open_files=len(process.open_files()) if hasattr(process, "open_files") else 0,
            threads=process.num_threads()
        )


@dataclass
class ResourceLimits:
    """Resource limits for the application."""
    max_memory_mb: float = 2048.0
    max_cpu_percent: float = 80.0
    max_open_files: int = 1000
    max_threads: int = 100
    max_analysis_time_seconds: float = 300.0


class ResourceTracker:
    """Tracks and manages resource usage."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        """Initialize resource tracker.
        
        Args:
            limits: Resource limits to enforce
        """
        self.limits = limits or ResourceLimits()
        self.snapshots: List[ResourceSnapshot] = []
        self.max_snapshots = 1000
        
        # Tracking state
        self.start_time = time.time()
        self.operation_resources: Dict[str, List[ResourceSnapshot]] = {}
        self.alerts: List[Dict[str, Any]] = []
        
        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_interval = 1.0  # seconds
        self._stop_monitoring = False
    
    async def start_monitoring(self):
        """Start background resource monitoring."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_loop())
            logger.info("Started resource monitoring")
    
    async def stop_monitoring(self):
        """Stop background resource monitoring."""
        self._stop_monitoring = True
        if self._monitoring_task:
            await self._monitoring_task
            self._monitoring_task = None
            logger.info("Stopped resource monitoring")
    
    async def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop_monitoring:
            try:
                snapshot = ResourceSnapshot.capture()
                self._add_snapshot(snapshot)
                self._check_limits(snapshot)
                await asyncio.sleep(self._monitoring_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {str(e)}")
                await asyncio.sleep(self._monitoring_interval)
    
    def _add_snapshot(self, snapshot: ResourceSnapshot):
        """Add a resource snapshot."""
        self.snapshots.append(snapshot)
        
        # Maintain max snapshots
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]
    
    def _check_limits(self, snapshot: ResourceSnapshot):
        """Check if resource limits are exceeded."""
        alerts = []
        
        if snapshot.memory_mb > self.limits.max_memory_mb:
            alerts.append({
                "type": "memory_limit",
                "severity": "high",
                "message": f"Memory usage ({snapshot.memory_mb:.1f}MB) exceeds limit ({self.limits.max_memory_mb}MB)",
                "value": snapshot.memory_mb,
                "limit": self.limits.max_memory_mb
            })
        
        if snapshot.cpu_percent > self.limits.max_cpu_percent:
            alerts.append({
                "type": "cpu_limit",
                "severity": "medium",
                "message": f"CPU usage ({snapshot.cpu_percent:.1f}%) exceeds limit ({self.limits.max_cpu_percent}%)",
                "value": snapshot.cpu_percent,
                "limit": self.limits.max_cpu_percent
            })
        
        if snapshot.open_files > self.limits.max_open_files:
            alerts.append({
                "type": "file_limit",
                "severity": "medium",
                "message": f"Open files ({snapshot.open_files}) exceeds limit ({self.limits.max_open_files})",
                "value": snapshot.open_files,
                "limit": self.limits.max_open_files
            })
        
        if snapshot.threads > self.limits.max_threads:
            alerts.append({
                "type": "thread_limit",
                "severity": "medium",
                "message": f"Thread count ({snapshot.threads}) exceeds limit ({self.limits.max_threads})",
                "value": snapshot.threads,
                "limit": self.limits.max_threads
            })
        
        # Add alerts with timestamp
        for alert in alerts:
            alert["timestamp"] = datetime.fromtimestamp(snapshot.timestamp).isoformat()
            self.alerts.append(alert)
            logger.warning(f"Resource alert: {alert['message']}")
    
    def track_operation(self, operation_name: str):
        """Context manager for tracking resource usage of an operation."""
        class OperationTracker:
            def __init__(self, tracker: ResourceTracker, name: str):
                self.tracker = tracker
                self.name = name
                self.start_snapshot = None
                self.end_snapshot = None
            
            def __enter__(self):
                self.start_snapshot = ResourceSnapshot.capture()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.end_snapshot = ResourceSnapshot.capture()
                
                # Store operation resources
                if self.name not in self.tracker.operation_resources:
                    self.tracker.operation_resources[self.name] = []
                
                self.tracker.operation_resources[self.name].append(self.start_snapshot)
                self.tracker.operation_resources[self.name].append(self.end_snapshot)
        
        return OperationTracker(self, operation_name)
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        snapshot = ResourceSnapshot.capture()
        
        return {
            "cpu_percent": snapshot.cpu_percent,
            "memory_mb": snapshot.memory_mb,
            "memory_percent": snapshot.memory_percent,
            "open_files": snapshot.open_files,
            "threads": snapshot.threads,
            "uptime_seconds": time.time() - self.start_time
        }
    
    def get_usage_trends(self, window_seconds: int = 60) -> Dict[str, Any]:
        """Get resource usage trends over a time window."""
        if not self.snapshots:
            return {}
        
        cutoff_time = time.time() - window_seconds
        recent_snapshots = [s for s in self.snapshots if s.timestamp > cutoff_time]
        
        if not recent_snapshots:
            return {}
        
        # Calculate trends
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        memory_values = [s.memory_mb for s in recent_snapshots]
        
        return {
            "window_seconds": window_seconds,
            "sample_count": len(recent_snapshots),
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values),
                "trend": "increasing" if cpu_values[-1] > cpu_values[0] else "decreasing"
            },
            "memory": {
                "avg": sum(memory_values) / len(memory_values),
                "min": min(memory_values),
                "max": max(memory_values),
                "trend": "increasing" if memory_values[-1] > memory_values[0] else "decreasing"
            }
        }
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get resource statistics for operations."""
        if operation_name:
            if operation_name not in self.operation_resources:
                return {}
            
            snapshots = self.operation_resources[operation_name]
            if len(snapshots) < 2:
                return {}
            
            # Pair up start/end snapshots
            operations = []
            for i in range(0, len(snapshots), 2):
                if i + 1 < len(snapshots):
                    start = snapshots[i]
                    end = snapshots[i + 1]
                    operations.append({
                        "duration": end.timestamp - start.timestamp,
                        "cpu_delta": end.cpu_percent - start.cpu_percent,
                        "memory_delta": end.memory_mb - start.memory_mb
                    })
            
            if not operations:
                return {}
            
            durations = [op["duration"] for op in operations]
            memory_deltas = [op["memory_delta"] for op in operations]
            
            return {
                "operation": operation_name,
                "count": len(operations),
                "duration": {
                    "avg": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations)
                },
                "memory_delta": {
                    "avg": sum(memory_deltas) / len(memory_deltas),
                    "min": min(memory_deltas),
                    "max": max(memory_deltas)
                }
            }
        else:
            # Return stats for all operations
            return {
                op_name: self.get_operation_stats(op_name)
                for op_name in self.operation_resources.keys()
            }
    
    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get resource alerts."""
        if severity:
            return [a for a in self.alerts if a["severity"] == severity]
        return self.alerts
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource usage summary."""
        current = self.get_current_usage()
        trends = self.get_usage_trends()
        
        return {
            "current": current,
            "trends": trends,
            "limits": {
                "memory_mb": self.limits.max_memory_mb,
                "cpu_percent": self.limits.max_cpu_percent,
                "open_files": self.limits.max_open_files,
                "threads": self.limits.max_threads
            },
            "alerts": {
                "total": len(self.alerts),
                "high": len([a for a in self.alerts if a["severity"] == "high"]),
                "medium": len([a for a in self.alerts if a["severity"] == "medium"]),
                "recent": self.alerts[-5:] if self.alerts else []
            },
            "operations": self.get_operation_stats()
        }


def track_resources(operation_name: str):
    """Decorator for tracking resource usage of functions."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracker = get_resource_tracker()
            with tracker.track_operation(operation_name):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracker = get_resource_tracker()
            with tracker.track_operation(operation_name):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Global resource tracker
_resource_tracker = ResourceTracker()


def get_resource_tracker() -> ResourceTracker:
    """Get the global resource tracker."""
    return _resource_tracker


# Auto-start monitoring if running as main process
if __name__ != "__main__":
    # Start monitoring in background when imported
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_resource_tracker.start_monitoring())
    except RuntimeError:
        # No event loop yet, will start monitoring later
        pass