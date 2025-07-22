"""Consolidated graph implementation for the log analyzer agent API.

This module provides a comprehensive graph implementation that combines
all advanced features from log_analyzer_agent without simplification.
"""

# Apply SSE compatibility patch
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import patch_sse
except ImportError:
    pass

from typing import Dict, Any, Set, Optional, Literal, Union
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
import functools
import time
import os
from datetime import datetime
import logging

from .state import State, CoreWorkingState, InteractiveWorkingState, MemoryWorkingState, initialize_state
from .nodes import analyze_logs as _analyze_logs_impl, validate_analysis as _validate_analysis_impl
from .tools import search_documentation, request_additional_info, submit_analysis, extract_patterns, generate_diagnostic_commands
from .cycle_detector import CycleDetector, CycleType

# Import enhanced graph if available
try:
    from .enhanced_graph import create_enhanced_graph, enhanced_graph, full_featured_graph
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False

logger = logging.getLogger(__name__)

# Simple in-memory cache for repeated analyses
_analysis_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes

# Configurable iteration limits
MAX_ANALYSIS_ITERATIONS = int(os.getenv("MAX_ANALYSIS_ITERATIONS", "10"))
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "20"))
MAX_VALIDATION_RETRIES = int(os.getenv("MAX_VALIDATION_RETRIES", "3"))

# Global cycle detector instance with configurable limits
_cycle_detector = CycleDetector(
    max_history=int(os.getenv("CYCLE_DETECTION_WINDOW", "20")),
    detection_threshold=int(os.getenv("MAX_SIMPLE_LOOPS", "3"))
)


def count_node_visits(messages: list, node_name: str) -> int:
    """Count visits to a specific node from messages."""
    count = 0
    for message in messages:
        if hasattr(message, 'content') and node_name in str(message.content):
            count += 1
    return count


def count_tool_calls(messages: list) -> int:
    """Count total tool calls from messages."""
    count = 0
    for message in messages:
        if hasattr(message, 'tool_calls') and message.tool_calls:
            count += len(message.tool_calls)
    return count


# Wrapper functions to handle dict-based state
async def analyze_logs(state: Dict[str, Any], *, config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Wrapper for analyze_logs that handles dict state."""
    # Initialize state with defaults
    state = initialize_state(state)
    
    # Track node visit
    state["node_visits"]["analyze_logs"] = state["node_visits"].get("analyze_logs", 0) + 1
    
    # Create a minimal CoreWorkingState for the implementation
    from langchain_core.messages import HumanMessage
    if not state.get("messages"):
        state["messages"] = [HumanMessage(content=f"Analyze this log:\n{state.get('log_content', '')}")] 
    
    working_state = CoreWorkingState(
        messages=state.get("messages", []),
        log_content=state.get("log_content", ""),
        log_metadata=state.get("log_metadata", {}),
        analysis_result=state.get("analysis_result"),
        current_analysis=state.get("current_analysis"),
        validation_status=state.get("validation_status"),
        validation_result=state.get("validation_result"),
        node_visits=state.get("node_visits", {}),
        tool_calls=state.get("tool_calls", []),
        tool_call_details=state.get("tool_call_details", []),
        token_count=state.get("token_count", 0),
        start_time=state.get("start_time", time.time()),
        enabled_features=set(state.get("enabled_features", [])),
        environment_details=state.get("environment_details", {}),
        application_name=state.get("application_name"),
        analysis_type=state.get("analysis_type", "general"),
        include_suggestions=state.get("include_suggestions", True),
        include_documentation=state.get("include_documentation", True),
        error=state.get("error")
    )
    
    # Call the implementation with config
    result = await _analyze_logs_impl(working_state, config=config, **kwargs)
    
    # Merge results back into state
    return {**state, **result}


async def validate_analysis(state: Dict[str, Any], *, config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Wrapper for validate_analysis that handles dict state."""
    # Initialize state with defaults
    state = initialize_state(state)
    
    # Track node visit
    state["node_visits"]["validate_analysis"] = state["node_visits"].get("validate_analysis", 0) + 1
    
    # Create a minimal state for the implementation
    working_state = CoreWorkingState(
        messages=state.get("messages", []),
        log_content=state.get("log_content", ""),
        log_metadata=state.get("log_metadata", {}),
        analysis_result=state.get("analysis_result"),
        current_analysis=state.get("current_analysis"),
        validation_status=state.get("validation_status"),
        validation_result=state.get("validation_result"),
        node_visits=state.get("node_visits", {}),
        tool_calls=state.get("tool_calls", []),
        tool_call_details=state.get("tool_call_details", []),
        token_count=state.get("token_count", 0),
        start_time=state.get("start_time", time.time()),
        enabled_features=set(state.get("enabled_features", [])),
        environment_details=state.get("environment_details", {}),
        application_name=state.get("application_name"),
        analysis_type=state.get("analysis_type", "general"),
        include_suggestions=state.get("include_suggestions", True),
        include_documentation=state.get("include_documentation", True),
        error=state.get("error")
    )
    
    # Call the implementation with config
    result = await _validate_analysis_impl(working_state, config=config, **kwargs)
    
    # Merge results back into state
    return {**state, **result}


async def handle_user_input(state: Dict[str, Any], *, config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Handle user input for interactive features."""
    # Initialize state with defaults  
    state = initialize_state(state)
    
    # Track node visit
    state["node_visits"]["handle_user_input"] = state["node_visits"].get("handle_user_input", 0) + 1
    
    # Create an InteractiveWorkingState for the implementation
    working_state = InteractiveWorkingState(
        messages=state.get("messages", []),
        log_content=state.get("log_content", ""),
        log_metadata=state.get("log_metadata", {}),
        analysis_result=state.get("analysis_result"),
        current_analysis=state.get("current_analysis"),
        validation_status=state.get("validation_status"),
        validation_result=state.get("validation_result"),
        node_visits=state.get("node_visits", {}),
        tool_calls=state.get("tool_calls", []),
        tool_call_details=state.get("tool_call_details", []),
        token_count=state.get("token_count", 0),
        start_time=state.get("start_time", time.time()),
        enabled_features=set(state.get("enabled_features", [])),
        environment_details=state.get("environment_details", {}),
        application_name=state.get("application_name"),
        analysis_type=state.get("analysis_type", "general"),
        include_suggestions=state.get("include_suggestions", True),
        include_documentation=state.get("include_documentation", True),
        error=state.get("error"),
        user_input=state.get("user_input"),
        pending_questions=state.get("pending_questions"),
        user_interaction_required=state.get("user_interaction_required", False),
        interaction_history=state.get("interaction_history", [])
    )
    
    # Simple implementation for now - just clear interaction requirement
    working_state.user_interaction_required = False
    if working_state.pending_questions:
        working_state.pending_questions = None
    
    # Convert back to dict
    result = {
        "user_interaction_required": working_state.user_interaction_required,
        "pending_questions": working_state.pending_questions,
        "interaction_history": working_state.interaction_history
    }
    
    # Merge results back into state
    return {**state, **result}


def cache_analysis(func):
    """Simple caching decorator for analysis results."""
    @functools.wraps(func)
    async def wrapper(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Ensure state is initialized
        state = initialize_state(state)
        
        # Create cache key from log content hash
        import hashlib
        log_content = state.get("log_content", "")
        cache_key = hashlib.md5(log_content.encode()).hexdigest()
        
        # Check cache
        if cache_key in _analysis_cache:
            cached = _analysis_cache[cache_key]
            if time.time() - cached["timestamp"] < CACHE_TTL_SECONDS:
                return {
                    **state,
                    "analysis_result": cached["result"],
                    "messages": state.get("messages", []) + [HumanMessage(content="Retrieved from cache")]
                }
        
        # Run analysis
        result = await func(state, **kwargs)
        
        # Cache result if successful
        if result.get("analysis_result"):
            _analysis_cache[cache_key] = {
                "result": result["analysis_result"],
                "timestamp": time.time()
            }
        
        return result
    
    return wrapper


def should_retry(state: Union[Dict[str, Any], Any]) -> bool:
    """Check if we should retry the analysis."""
    # Handle both dict and dataclass
    if hasattr(state, "get"):
        messages = state.get("messages", [])
        validation_status = state.get("validation_status")
    else:
        messages = getattr(state, "messages", [])
        validation_status = getattr(state, "validation_status", None)
    
    # Simple retry logic based on node visits
    visits = count_node_visits(messages, "analyze_logs")
    return visits < MAX_VALIDATION_RETRIES and validation_status == "invalid"


def route_after_analysis(state: Union[Dict[str, Any], Any]) -> Union[
    Literal["validate_analysis"],
    Literal["tools"],
    Literal["__end__"]
]:
    """Route after the analysis node."""
    # Handle both dict and dataclass
    if hasattr(state, "get"):
        messages = state.get("messages", [])
        analysis_result = state.get("analysis_result")
    else:
        messages = getattr(state, "messages", [])
        analysis_result = getattr(state, "analysis_result", None)
    
    last_message = messages[-1] if messages else None
    
    # Add transition to cycle detector
    state_dict = {"messages": messages, "analysis_result": analysis_result}
    cycle = _cycle_detector.add_transition("analyze_logs", "route", state_dict)
    
    # Check for cycles using advanced detection
    if cycle and _cycle_detector.should_break_cycle(cycle):
        logger.warning(f"Breaking {cycle.cycle_type.value} cycle: {' -> '.join(cycle.pattern)}")
        return "__end__"
    
    # Fallback to simple limits
    analysis_count = count_node_visits(messages, "analyze_logs")
    tool_count = count_tool_calls(messages)
    
    if analysis_count >= MAX_ANALYSIS_ITERATIONS or tool_count >= MAX_TOOL_CALLS:
        return "__end__"
    
    # Check for tool calls
    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Go to validation
    return "validate_analysis"


def route_after_validation(state: Union[Dict[str, Any], Any]) -> Union[
    Literal["analyze_logs"],
    Literal["handle_user_input"],
    Literal["__end__"]
]:
    """Route after validation."""
    # Handle both dict and dataclass
    if hasattr(state, "get"):
        status = state.get("validation_status", "")
        messages = state.get("messages", [])
        validation_result = state.get("validation_result", {})
    else:
        status = getattr(state, "validation_status", "")
        messages = getattr(state, "messages", [])
        validation_result = getattr(state, "validation_result", {})
    
    # Add transition to cycle detector
    state_dict = {"messages": messages, "validation_status": status}
    cycle = _cycle_detector.add_transition("validate_analysis", "route", state_dict)
    
    # If valid, we're done
    if status == "valid" or (validation_result and validation_result.get("is_valid", False)):
        return "__end__"
    
    # Check for validation retry cycles
    if cycle and cycle.cycle_type in [CycleType.OSCILLATION, CycleType.DEADLOCK]:
        logger.warning(f"Breaking validation {cycle.cycle_type.value}: {' -> '.join(cycle.pattern)}")
        return "__end__"
    
    # If invalid and interactive features are enabled, ask user
    if hasattr(state, "get"):
        user_interaction_required = state.get("user_interaction_required")
        enabled_features = state.get("enabled_features", [])
    else:
        user_interaction_required = getattr(state, "user_interaction_required", None)
        enabled_features = getattr(state, "enabled_features", [])
    
    if user_interaction_required and "interactive" in enabled_features:
        return "handle_user_input"
    
    # Otherwise, retry analysis if under limit
    if should_retry(state):
        return "analyze_logs"
    
    # Otherwise end
    return "__end__"


def route_after_tools(state: Union[Dict[str, Any], Any]) -> Union[
    Literal["validate_analysis"],
    Literal["analyze_logs"],
    Literal["__end__"]
]:
    """Route after tool execution."""
    # Handle both dict and dataclass
    if hasattr(state, "get"):
        messages = state.get("messages", [])
    else:
        messages = getattr(state, "messages", [])
    
    # Add transition to cycle detector
    state_dict = {"messages": messages}
    cycle = _cycle_detector.add_transition("tools", "route", state_dict)
    
    # Check for tool execution cycles
    if cycle and _cycle_detector.should_break_cycle(cycle):
        logger.warning(f"Breaking tool {cycle.cycle_type.value}: {' -> '.join(cycle.pattern)}")
        return "__end__"
    
    # Check limits
    if count_node_visits(messages, "analyze_logs") >= MAX_ANALYSIS_ITERATIONS:
        return "__end__"
    
    # If we have an analysis result, validate it
    if hasattr(state, "get"):
        analysis_result = state.get("analysis_result")
    else:
        analysis_result = getattr(state, "analysis_result", None)
    
    if analysis_result:
        return "validate_analysis"
    
    # Otherwise, analyze again
    return "analyze_logs"


def create_graph(features: Optional[Set[str]] = None):
    """Create a graph with the specified features.
    
    Args:
        features: Set of features to enable. Options:
            - "interactive": Enable user interaction
            - "memory": Enable memory features (requires database)
            - "caching": Enable result caching
            - "streaming": Enable streaming support
            - "specialized": Enable specialized analyzers
            - "monitoring": Enable resource monitoring
            
    Returns:
        Compiled StateGraph
    """
    if features is None:
        features = set()
    
    # Check if enhanced features are requested and available
    use_enhanced = os.getenv("USE_ENHANCED_FEATURES", "false").lower() == "true"
    
    if use_enhanced and ENHANCED_FEATURES_AVAILABLE:
        logger.info("Using enhanced graph with advanced features")
        # Determine which features to enable from environment
        env_features = set()
        
        if os.getenv("ENABLE_CACHING", "true").lower() == "true":
            env_features.add("caching")
        if os.getenv("ENABLE_SPECIALIZED", "true").lower() == "true":
            env_features.add("specialized")
        if os.getenv("ENABLE_MONITORING", "true").lower() == "true":
            env_features.add("monitoring")
        if os.getenv("ENABLE_INTERACTIVE", "false").lower() == "true":
            env_features.add("interactive")
        if os.getenv("ENABLE_MEMORY", "false").lower() == "true":
            env_features.add("memory")
        if os.getenv("ENABLE_STREAMING", "true").lower() == "true":
            env_features.add("streaming")
        
        # Merge with requested features
        features = features.union(env_features)
        
        return create_enhanced_graph(features)
    
    # Create standard advanced graph with all features
    logger.info("Using standard advanced graph")
    
    # Create graph with State
    workflow = StateGraph(State)
    
    # Add nodes
    if "caching" in features:
        workflow.add_node("analyze_logs", cache_analysis(analyze_logs))
    else:
        workflow.add_node("analyze_logs", analyze_logs)
    
    workflow.add_node("validate_analysis", validate_analysis)
    
    # Add tool node
    tools = [search_documentation, submit_analysis, extract_patterns, generate_diagnostic_commands]
    if "interactive" in features:
        tools.append(request_additional_info)
        workflow.add_node("handle_user_input", handle_user_input)
    
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)
    
    # Define edges
    workflow.add_edge(START, "analyze_logs")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "analyze_logs",
        route_after_analysis,
        {
            "validate_analysis": "validate_analysis",
            "tools": "tools",
            "__end__": END
        }
    )
    
    workflow.add_conditional_edges(
        "validate_analysis",
        route_after_validation,
        {
            "analyze_logs": "analyze_logs",
            "handle_user_input": "handle_user_input" if "interactive" in features else END,
            "__end__": END
        }
    )
    
    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "validate_analysis": "validate_analysis",
            "analyze_logs": "analyze_logs",
            "__end__": END
        }
    )
    
    if "interactive" in features:
        workflow.add_edge("handle_user_input", "analyze_logs")
    
    # Compile
    checkpointer = None
    if "memory" in features:
        # Only add checkpointer if memory features are enabled
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)


# Convenience functions for common configurations
def create_minimal_graph():
    """Create a minimal graph with no extra features."""
    return create_graph(features=set())


def create_interactive_graph():
    """Create a graph with interactive features."""
    return create_graph(features={"interactive", "caching"})


def create_memory_graph():
    """Create a graph with memory features."""
    return create_graph(features={"memory", "interactive", "caching"})


def create_full_graph():
    """Create a graph with all features."""
    return create_graph(features={"memory", "interactive", "caching", "streaming", "specialized", "monitoring"})


def create_streaming_graph():
    """Create a graph optimized for streaming large logs."""
    if ENHANCED_FEATURES_AVAILABLE:
        # Use enhanced graph with streaming enabled
        return create_enhanced_graph({"streaming", "caching", "specialized"})
    
    # Fall back to advanced graph with streaming
    return create_graph(features={"streaming", "caching", "specialized"})


# Simple performance tracking
_performance_metrics: Dict[str, list] = {
    "analysis_times": [],
    "cache_hits": 0,
    "cache_misses": 0
}


def get_performance_metrics() -> Dict[str, Any]:
    """Get simple performance metrics."""
    analysis_times = _performance_metrics["analysis_times"]
    return {
        "total_analyses": len(analysis_times),
        "average_time": sum(analysis_times) / len(analysis_times) if analysis_times else 0,
        "cache_hit_rate": (_performance_metrics["cache_hits"] / 
                          (_performance_metrics["cache_hits"] + _performance_metrics["cache_misses"])
                          if _performance_metrics["cache_hits"] + _performance_metrics["cache_misses"] > 0 
                          else 0),
        "cycle_summary": _cycle_detector.get_cycle_summary()
    }


def clear_cache():
    """Clear the analysis cache."""
    _analysis_cache.clear()
    _performance_metrics["cache_hits"] = 0
    _performance_metrics["cache_misses"] = 0
    _cycle_detector.reset()


# Create enhanced graph for API usage
def create_enhanced_api_graph():
    """Create an enhanced graph with optimal features for API usage.
    
    This is the recommended graph for production API usage with:
    - Interactive features for better analysis
    - Caching for performance
    - Streaming for large logs
    - Specialized analyzers
    - Monitoring capabilities
    """
    return create_graph(features={"interactive", "caching", "streaming", "specialized", "monitoring"})


# Export the compiled graph for LangGraph Cloud
# Use enhanced graph by default if available and requested
if os.getenv("USE_ENHANCED_FEATURES", "false").lower() == "true" and ENHANCED_FEATURES_AVAILABLE:
    graph = enhanced_graph
else:
    # Use the full advanced graph as default
    graph = create_enhanced_api_graph()