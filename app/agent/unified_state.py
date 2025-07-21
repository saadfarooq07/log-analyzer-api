"""Unified state management for the log analyzer API.

This module provides a unified state class that consolidates all state management
and enables feature composition similar to the improved log_analyzer_agent.
"""

from typing import Dict, Any, List, Set, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import time
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


@dataclass
class UnifiedState:
    """Unified state class that supports all features through composition.
    
    This replaces the simple State TypedDict with a more powerful implementation
    that supports interactive mode, memory, streaming, and other advanced features.
    """
    
    # Core fields (always present)
    messages: List[BaseMessage] = field(default_factory=list)
    log_content: str = ""
    log_metadata: Dict[str, Any] = field(default_factory=dict)
    analysis_result: Optional[Dict[str, Any]] = None
    validation_status: Optional[str] = None
    current_analysis: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Tracking fields
    node_visits: Dict[str, int] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    token_count: int = 0
    start_time: float = field(default_factory=time.time)
    
    # Feature flags
    enabled_features: Set[str] = field(default_factory=set)
    
    # Interactive mode fields
    user_interaction_required: bool = False
    pending_questions: List[Dict[str, Any]] = field(default_factory=list)
    user_responses: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Memory/persistence fields
    thread_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    analysis_history: List[Dict[str, Any]] = field(default_factory=list)
    memory_matches: List[Dict[str, Any]] = field(default_factory=list)
    application_context: Dict[str, Any] = field(default_factory=dict)
    
    # Streaming fields
    is_streaming: bool = False
    current_chunk_index: int = 0
    total_chunks: int = 0
    chunk_results: List[Dict[str, Any]] = field(default_factory=list)
    stream_buffer: List[str] = field(default_factory=list)
    
    # Resource tracking
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    api_calls_count: Dict[str, int] = field(default_factory=dict)
    rate_limit_remaining: Dict[str, int] = field(default_factory=dict)
    
    # Cycle detection
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    cycle_detection_enabled: bool = True
    detected_cycles: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    node_execution_times: Dict[str, List[float]] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    
    def enable_feature(self, feature: str) -> None:
        """Enable a specific feature."""
        self.enabled_features.add(feature)
        
    def disable_feature(self, feature: str) -> None:
        """Disable a specific feature."""
        self.enabled_features.discard(feature)
        
    def has_feature(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return feature in self.enabled_features
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for LangGraph compatibility."""
        # Convert messages to serializable format
        messages_data = []
        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                messages_data.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages_data.append({"type": "ai", "content": msg.content})
            else:
                messages_data.append({"type": "system", "content": str(msg)})
        
        return {
            # Core fields
            "messages": messages_data,
            "log_content": self.log_content,
            "log_metadata": self.log_metadata,
            "analysis_result": self.analysis_result,
            "validation_status": self.validation_status,
            "current_analysis": self.current_analysis,
            "validation_result": self.validation_result,
            "error": self.error,
            
            # Tracking fields
            "node_visits": self.node_visits,
            "tool_calls": self.tool_calls,
            "token_count": self.token_count,
            "start_time": self.start_time,
            
            # Feature flags
            "enabled_features": list(self.enabled_features),
            
            # Interactive mode fields (only if enabled)
            **({"user_interaction_required": self.user_interaction_required,
                "pending_questions": self.pending_questions,
                "user_responses": self.user_responses,
                "interaction_history": self.interaction_history} 
               if self.has_feature("interactive") else {}),
            
            # Memory fields (only if enabled)
            **({"thread_id": self.thread_id,
                "checkpoint_id": self.checkpoint_id,
                "analysis_history": self.analysis_history,
                "memory_matches": self.memory_matches,
                "application_context": self.application_context}
               if self.has_feature("memory") else {}),
            
            # Streaming fields (only if enabled)
            **({"is_streaming": self.is_streaming,
                "current_chunk_index": self.current_chunk_index,
                "total_chunks": self.total_chunks,
                "chunk_results": self.chunk_results}
               if self.has_feature("streaming") else {}),
            
            # Resource tracking (only if enabled)
            **({"memory_usage_mb": self.memory_usage_mb,
                "cpu_usage_percent": self.cpu_usage_percent,
                "api_calls_count": self.api_calls_count,
                "rate_limit_remaining": self.rate_limit_remaining}
               if self.has_feature("resource_tracking") else {}),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedState":
        """Create UnifiedState from dictionary."""
        # Reconstruct messages
        messages = []
        for msg_data in data.get("messages", []):
            if msg_data["type"] == "human":
                messages.append(HumanMessage(content=msg_data["content"]))
            elif msg_data["type"] == "ai":
                messages.append(AIMessage(content=msg_data["content"]))
        
        state = cls(
            messages=messages,
            log_content=data.get("log_content", ""),
            log_metadata=data.get("log_metadata", {}),
            analysis_result=data.get("analysis_result"),
            validation_status=data.get("validation_status"),
            current_analysis=data.get("current_analysis"),
            validation_result=data.get("validation_result"),
            error=data.get("error"),
            node_visits=data.get("node_visits", {}),
            tool_calls=data.get("tool_calls", []),
            token_count=data.get("token_count", 0),
            start_time=data.get("start_time", time.time()),
            enabled_features=set(data.get("enabled_features", []))
        )
        
        # Set feature-specific fields
        if "interactive" in state.enabled_features:
            state.user_interaction_required = data.get("user_interaction_required", False)
            state.pending_questions = data.get("pending_questions", [])
            state.user_responses = data.get("user_responses", {})
            state.interaction_history = data.get("interaction_history", [])
        
        if "memory" in state.enabled_features:
            state.thread_id = data.get("thread_id")
            state.checkpoint_id = data.get("checkpoint_id")
            state.analysis_history = data.get("analysis_history", [])
            state.memory_matches = data.get("memory_matches", [])
            state.application_context = data.get("application_context", {})
        
        if "streaming" in state.enabled_features:
            state.is_streaming = data.get("is_streaming", False)
            state.current_chunk_index = data.get("current_chunk_index", 0)
            state.total_chunks = data.get("total_chunks", 0)
            state.chunk_results = data.get("chunk_results", [])
        
        if "resource_tracking" in state.enabled_features:
            state.memory_usage_mb = data.get("memory_usage_mb", 0.0)
            state.cpu_usage_percent = data.get("cpu_usage_percent", 0.0)
            state.api_calls_count = data.get("api_calls_count", {})
            state.rate_limit_remaining = data.get("rate_limit_remaining", {})
        
        return state
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message and update token count."""
        self.messages.append(message)
        # Simple token estimation (4 chars = 1 token)
        self.token_count += len(str(message.content)) // 4
    
    def increment_node_visit(self, node_name: str) -> None:
        """Track node visits for cycle detection."""
        self.node_visits[node_name] = self.node_visits.get(node_name, 0) + 1
    
    def add_tool_call(self, tool_name: str, args: Dict[str, Any], result: Any) -> None:
        """Track tool calls."""
        self.tool_calls.append({
            "tool": tool_name,
            "args": args,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
    
    def update_resource_usage(self, memory_mb: float, cpu_percent: float) -> None:
        """Update resource tracking metrics."""
        self.memory_usage_mb = memory_mb
        self.cpu_usage_percent = cpu_percent
    
    def record_api_call(self, api_name: str) -> None:
        """Record an API call for rate limiting."""
        self.api_calls_count[api_name] = self.api_calls_count.get(api_name, 0) + 1
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since analysis started."""
        return time.time() - self.start_time


def create_unified_state(
    log_content: str,
    features: Optional[Set[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> UnifiedState:
    """Create a UnifiedState with specified features enabled.
    
    Args:
        log_content: The log content to analyze
        features: Set of features to enable
        metadata: Additional metadata
        **kwargs: Additional state fields
        
    Returns:
        Configured UnifiedState instance
    """
    features = features or set()
    
    # Auto-enable streaming for large logs
    log_size_mb = len(log_content.encode()) / (1024 * 1024)
    if log_size_mb > 10:
        features.add("streaming")
    
    state = UnifiedState(
        log_content=log_content,
        log_metadata=metadata or {},
        enabled_features=features,
        **kwargs
    )
    
    # Initialize feature-specific fields
    if "streaming" in features:
        # Calculate chunks
        lines = log_content.splitlines()
        chunk_size = 1000  # lines per chunk
        state.total_chunks = (len(lines) + chunk_size - 1) // chunk_size
        state.is_streaming = True
    
    return state