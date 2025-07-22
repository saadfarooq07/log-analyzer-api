"""Consolidated state management for the log analyzer agent API.

This module implements the full state management system from log_analyzer_agent
with progressive enhancement, feature flags, and advanced state tracking.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Set, Annotated, Sequence, TypedDict, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json

from langchain_core.messages import BaseMessage, AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class StateFeature(str, Enum):
    """Available state features for progressive enhancement."""
    INTERACTIVE = "interactive"
    STREAMING = "streaming"
    MEMORY = "memory"
    CHECKPOINTING = "checkpointing"
    RETRY = "retry"
    VALIDATION = "validation"
    CACHING = "caching"
    SPECIALIZED = "specialized"
    MONITORING = "monitoring"


# ============================================================================
# INPUT STATE (Immutable)
# ============================================================================

@dataclass(frozen=True)
class InputState:
    """Immutable input state representing user request.
    
    This state is created once and never modified during execution.
    """
    log_content: str
    environment_details: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    application_name: Optional[str] = None
    session_id: Optional[str] = None
    analysis_type: str = "general"
    include_suggestions: bool = True
    include_documentation: bool = True
    requested_features: Set[StateFeature] = field(default_factory=set)
    
    def __post_init__(self):
        """Validate input state and auto-detect features."""
        if not self.log_content:
            raise ValueError("log_content cannot be empty")
        
        # Auto-detect features based on log size
        if len(self.log_content) > 10 * 1024 * 1024:  # 10MB
            object.__setattr__(self, 'requested_features', 
                             self.requested_features | {StateFeature.STREAMING})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['requested_features'] = list(self.requested_features)
        return data


# ============================================================================
# WORKING STATE (Mutable, with Progressive Enhancement)
# ============================================================================

@dataclass
class CoreWorkingState:
    """Minimal working state for basic log analysis.
    
    This is the base state that all other working states extend.
    """
    # Core fields (always present)
    log_content: str = ""
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    log_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis state
    analysis_result: Optional[Dict[str, Any]] = None
    current_analysis: Optional[Dict[str, Any]] = None
    validation_status: Optional[str] = None
    validation_result: Optional[Dict[str, Any]] = None
    
    # Execution tracking
    node_visits: Dict[str, int] = field(default_factory=dict)
    tool_calls: List[str] = field(default_factory=list)
    tool_call_details: List[Dict[str, Any]] = field(default_factory=list)
    token_count: int = 0
    start_time: float = field(default_factory=lambda: datetime.now().timestamp())
    
    # Feature flags
    enabled_features: Set[StateFeature] = field(default_factory=set)
    
    # Input fields for compatibility
    environment_details: Dict[str, Any] = field(default_factory=dict)
    application_name: Optional[str] = None
    analysis_type: str = "general"
    include_suggestions: bool = True
    include_documentation: bool = True
    
    # Error handling
    error: Optional[str] = None
    
    def increment_node_visit(self, node_name: str) -> None:
        """Track node execution for cycle prevention."""
        self.node_visits[node_name] = self.node_visits.get(node_name, 0) + 1
    
    def add_tool_call(self, tool_name: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Track tool usage."""
        self.tool_calls.append(tool_name)
        if details:
            self.tool_call_details.append({
                "tool": tool_name,
                "timestamp": datetime.now().isoformat(),
                **details
            })
    
    def has_feature(self, feature: StateFeature) -> bool:
        """Check if a feature is enabled."""
        return feature in self.enabled_features
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution state."""
        return {
            "node_visits": self.node_visits,
            "total_tool_calls": len(self.tool_calls),
            "unique_tools_used": len(set(self.tool_calls)),
            "token_count": self.token_count,
            "elapsed_time": datetime.now().timestamp() - self.start_time,
            "enabled_features": list(self.enabled_features)
        }


@dataclass
class InteractiveWorkingState(CoreWorkingState):
    """Working state with user interaction support."""
    user_input: Optional[str] = None
    pending_questions: Optional[List[Dict[str, str]]] = None
    user_interaction_required: bool = False
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_interaction(self, question: str, answer: Optional[str] = None) -> None:
        """Record user interaction."""
        self.interaction_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer
        })


@dataclass
class StreamingWorkingState(InteractiveWorkingState):
    """Working state with streaming and chunking support."""
    # Streaming support
    is_streaming: bool = False
    current_chunk_index: int = 0
    total_chunks: int = 0
    chunk_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Checkpointing support
    checkpoint_identifier: Optional[str] = None
    last_checkpoint: Optional[datetime] = None
    checkpoint_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_chunk_result(self, chunk_index: int, result: Dict[str, Any]) -> None:
        """Add result from processing a chunk."""
        self.chunk_results.append({
            "index": chunk_index,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        self.current_chunk_index = chunk_index + 1
    
    def create_checkpoint(self) -> str:
        """Create a checkpoint and return its ID."""
        from uuid import uuid4
        self.checkpoint_identifier = f"ckpt_{uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.last_checkpoint = datetime.now()
        return self.checkpoint_identifier


@dataclass
class MemoryWorkingState(StreamingWorkingState):
    """Full-featured working state with memory and persistence."""
    # Memory features
    memory_matches: Optional[List[Dict[str, Any]]] = None
    application_context: Optional[Dict[str, Any]] = None
    user_context: Optional[Dict[str, Any]] = None
    
    # Session management
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    
    # Persistence metadata
    last_saved: Optional[datetime] = None
    save_count: int = 0
    
    def add_memory_match(self, match: Dict[str, Any]) -> None:
        """Add a relevant memory match."""
        if self.memory_matches is None:
            self.memory_matches = []
        self.memory_matches.append({
            **match,
            "retrieved_at": datetime.now().isoformat()
        })
    
    def mark_saved(self) -> None:
        """Mark state as saved to persistence layer."""
        self.last_saved = datetime.now()
        self.save_count += 1


# ============================================================================
# OUTPUT STATE (Immutable)
# ============================================================================

@dataclass(frozen=True)
class OutputState:
    """Immutable output state representing analysis results.
    
    This state is created at the end of processing and contains final results.
    """
    # Analysis results
    issues: List[Dict[str, Any]]
    root_cause: str
    recommendations: List[str]
    documentation_references: List[Dict[str, str]]
    diagnostic_commands: List[Dict[str, str]]
    confidence_score: float
    
    # Execution metadata
    execution_summary: Dict[str, Any]
    processing_time: float
    features_used: List[str]
    
    # Optional enhanced results
    memory_context: Optional[Dict[str, Any]] = None
    chunk_summaries: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


# ============================================================================
# TYPED DICT VERSIONS (for LangGraph compatibility)
# ============================================================================

class State(TypedDict, total=False):
    """Main TypedDict state for LangGraph compatibility."""
    # Core fields
    messages: List[BaseMessage]
    log_content: str
    log_metadata: Dict[str, Any]
    
    # Analysis fields
    analysis_result: Optional[Dict[str, Any]]
    current_analysis: Optional[Dict[str, Any]]
    validation_status: Optional[str]
    validation_result: Optional[Dict[str, Any]]
    
    # Tracking fields
    node_visits: Dict[str, int]
    tool_calls: List[str]
    tool_call_details: List[Dict[str, Any]]
    token_count: int
    start_time: float
    
    # Feature flags
    enabled_features: List[str]
    
    # Input fields
    environment_details: Dict[str, Any]
    application_name: Optional[str]
    analysis_type: str
    include_suggestions: bool
    include_documentation: bool
    
    # Interactive fields
    user_input: Optional[str]
    pending_questions: Optional[List[Dict[str, str]]]
    user_interaction_required: bool
    interaction_history: List[Dict[str, Any]]
    
    # Streaming fields
    is_streaming: bool
    current_chunk_index: int
    total_chunks: int
    chunk_results: List[Dict[str, Any]]
    checkpoint_identifier: Optional[str]
    last_checkpoint: Optional[str]
    checkpoint_metadata: Dict[str, Any]
    
    # Memory fields
    memory_matches: Optional[List[Dict[str, Any]]]
    application_context: Optional[Dict[str, Any]]
    user_context: Optional[Dict[str, Any]]
    session_id: Optional[str]
    thread_id: Optional[str]
    last_saved: Optional[str]
    save_count: int
    
    # Error handling
    error: Optional[str]


# ============================================================================
# STATE UTILITIES
# ============================================================================

def create_working_state(
    input_state: InputState,
    messages: Optional[List[AnyMessage]] = None
) -> Union[CoreWorkingState, InteractiveWorkingState, StreamingWorkingState, MemoryWorkingState]:
    """Create appropriate working state based on requested features.
    
    This factory function creates the right state class based on features.
    """
    if messages is None:
        messages = []
    
    base_kwargs = {
        "messages": messages,
        "log_content": input_state.log_content,
        "log_metadata": {
            "size_bytes": len(input_state.log_content),
            "user_id": input_state.user_id,
            "application_name": input_state.application_name,
            "session_id": input_state.session_id
        },
        "enabled_features": input_state.requested_features,
        "environment_details": input_state.environment_details or {},
        "application_name": input_state.application_name,
        "analysis_type": input_state.analysis_type,
        "include_suggestions": input_state.include_suggestions,
        "include_documentation": input_state.include_documentation
    }
    
    # Progressive enhancement based on features
    if StateFeature.MEMORY in input_state.requested_features:
        return MemoryWorkingState(**base_kwargs)
    elif StateFeature.STREAMING in input_state.requested_features:
        return StreamingWorkingState(**base_kwargs)
    elif StateFeature.INTERACTIVE in input_state.requested_features:
        return InteractiveWorkingState(**base_kwargs)
    else:
        return CoreWorkingState(**base_kwargs)


def working_to_output(
    working_state: CoreWorkingState,
    processing_start_time: float
) -> OutputState:
    """Convert working state to immutable output state."""
    # Extract analysis result
    result = working_state.analysis_result or {}
    
    # Build output state
    return OutputState(
        issues=result.get("issues", []),
        root_cause=result.get("root_cause", "Unknown"),
        recommendations=result.get("recommendations", []),
        documentation_references=result.get("documentation_references", []),
        diagnostic_commands=result.get("diagnostic_commands", []),
        confidence_score=result.get("confidence_score", 0.0),
        execution_summary=working_state.get_execution_summary(),
        processing_time=datetime.now().timestamp() - processing_start_time,
        features_used=list(working_state.enabled_features),
        memory_context=getattr(working_state, "memory_context", None),
        chunk_summaries=getattr(working_state, "chunk_results", None)
    )


def initialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize state with default values for missing fields."""
    # Ensure messages field exists
    if "messages" not in state:
        state["messages"] = []
    
    # Initialize tracking fields
    if "node_visits" not in state:
        state["node_visits"] = {}
    if "tool_calls" not in state:
        state["tool_calls"] = []
    if "tool_call_details" not in state:
        state["tool_call_details"] = []
    if "token_count" not in state:
        state["token_count"] = 0
    if "start_time" not in state:
        state["start_time"] = datetime.now().timestamp()
    
    # Initialize metadata
    if "log_metadata" not in state:
        state["log_metadata"] = {}
    
    # Initialize features
    if "enabled_features" not in state:
        state["enabled_features"] = []
    
    # Initialize input fields
    if "environment_details" not in state:
        state["environment_details"] = {}
    if "analysis_type" not in state:
        state["analysis_type"] = "general"
    if "include_suggestions" not in state:
        state["include_suggestions"] = True
    if "include_documentation" not in state:
        state["include_documentation"] = True
    
    # Initialize interactive fields
    if "user_interaction_required" not in state:
        state["user_interaction_required"] = False
    if "interaction_history" not in state:
        state["interaction_history"] = []
    
    # Initialize streaming fields
    if "is_streaming" not in state:
        state["is_streaming"] = False
    if "current_chunk_index" not in state:
        state["current_chunk_index"] = 0
    if "total_chunks" not in state:
        state["total_chunks"] = 0
    if "chunk_results" not in state:
        state["chunk_results"] = []
    if "checkpoint_metadata" not in state:
        state["checkpoint_metadata"] = {}
    
    # Initialize memory fields
    if "save_count" not in state:
        state["save_count"] = 0
    
    return state


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# Aliases for backward compatibility
CoreState = CoreWorkingState
InteractiveState = InteractiveWorkingState
MemoryState = MemoryWorkingState

# Legacy function names
def create_state_class(enable_memory: bool = False, 
                      enable_interactive: bool = False) -> type:
    """Legacy factory function for backward compatibility."""
    if enable_memory:
        return MemoryWorkingState
    elif enable_interactive:
        return InteractiveWorkingState
    else:
        return CoreWorkingState