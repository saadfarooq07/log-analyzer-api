"""Enhanced graph implementation with all advanced features.

This module creates the enhanced LangGraph workflow that includes all the
improved features from the log_analyzer_agent implementation.
"""

import logging
import uuid
from typing import Dict, Any, Set, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
import asyncio

from .state import State
from .unified_state import UnifiedState, create_unified_state
from .nodes import analyze_logs as base_analyze_logs, validate_analysis
from .tools import search_documentation, extract_patterns, generate_diagnostic_commands
from .cycle_detector import CycleDetector, CycleType
from .circuit_breaker import get_circuit_breaker
from .rate_limiter import get_rate_limiter
from .specialized_analyzers import get_specialized_analyzer_manager
from .streaming_processor import get_streaming_analyzer
from .cache_manager import get_cache_manager, cached_operation
from .resource_tracker import get_resource_tracker, track_resources
from .interactive_handler import get_interactive_handler
from .memory_manager import get_memory_manager

logger = logging.getLogger(__name__)

# Global cycle detector
_cycle_detector = CycleDetector(max_history=20, detection_threshold=3)


async def enhanced_analyze_logs(state: State) -> Dict[str, Any]:
    """Enhanced analysis node with all advanced features."""
    # Convert to UnifiedState for advanced features
    unified_state = create_unified_state(
        log_content=state["log_content"],
        features=set(state.get("enabled_features", [])),
        metadata=state.get("environment_details", {})
    )
    
    # Copy relevant fields
    unified_state.messages = state.get("messages", [])
    unified_state.analysis_result = state.get("analysis_result")
    unified_state.validation_status = state.get("validation_status")
    
    # Track resource usage
    resource_tracker = get_resource_tracker()
    with resource_tracker.track_operation("analyze_logs"):
        
        # Check cache first
        cache_manager = get_cache_manager()
        cached_result = cache_manager.get_analysis(
            unified_state.log_content,
            unified_state.log_metadata
        )
        
        if cached_result:
            logger.info("Using cached analysis result")
            unified_state.cache_hits += 1
            state["current_analysis"] = cached_result
            state["messages"] = unified_state.messages
            return state
        
        unified_state.cache_misses += 1
        
        # Apply rate limiting
        rate_limiter = get_rate_limiter("gemini")
        await rate_limiter.acquire()
        
        # Apply circuit breaker
        circuit_breaker = get_circuit_breaker("analysis", failure_threshold=3)
        
        try:
            # Check if streaming is needed
            log_size_mb = len(unified_state.log_content.encode()) / (1024 * 1024)
            
            if log_size_mb > 10 or unified_state.has_feature("streaming"):
                # Use streaming analyzer
                streaming_analyzer = get_streaming_analyzer()
                result = await circuit_breaker.async_call(
                    streaming_analyzer.analyze_with_streaming,
                    unified_state.log_content,
                    unified_state,
                    base_analyze_logs
                )
            else:
                # Regular analysis
                result = await circuit_breaker.async_call(
                    base_analyze_logs,
                    state
                )
            
            # Apply specialized analyzers
            if unified_state.has_feature("specialized"):
                analyzer_manager = get_specialized_analyzer_manager()
                result = analyzer_manager.analyze(
                    unified_state.log_content,
                    result.get("current_analysis", result)
                )
            
            # Cache the result
            cache_manager.cache_analysis(
                unified_state.log_content,
                result,
                unified_state.log_metadata
            )
            
            # Save to memory if enabled
            if unified_state.has_feature("memory"):
                memory_manager = get_memory_manager()
                analysis_id = str(uuid.uuid4())
                await memory_manager.create_analysis_record(
                    analysis_id,
                    unified_state.thread_id,
                    unified_state.log_content,
                    unified_state.log_metadata
                )
                unified_state.checkpoint_id = analysis_id
            
            # Generate interactive questions if needed
            if unified_state.has_feature("interactive") and result.get("current_analysis"):
                interactive_handler = get_interactive_handler()
                questions = interactive_handler.generate_clarification_questions(
                    unified_state,
                    result["current_analysis"]
                )
                if questions:
                    interactive_handler.add_questions_to_state(unified_state, questions)
                    state["user_interaction_required"] = True
                    state["pending_questions"] = unified_state.pending_questions
            
            # Update state
            state["current_analysis"] = result.get("current_analysis", result)
            state["messages"] = unified_state.messages
            state["node_visits"] = unified_state.node_visits
            
            # Track node visit
            unified_state.increment_node_visit("analyze_logs")
            
            return state
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            state["error"] = str(e)
            return state


async def handle_user_input(state: State) -> Dict[str, Any]:
    """Handle interactive user input."""
    if not state.get("user_interaction_required"):
        return state
    
    interactive_handler = get_interactive_handler()
    unified_state = UnifiedState.from_dict(state)
    
    # Process any provided responses
    if state.get("user_input"):
        for question_id, answer in state["user_input"].items():
            interactive_handler.process_user_response(
                unified_state,
                question_id,
                answer
            )
    
    # Update state
    state["pending_questions"] = unified_state.pending_questions
    state["user_responses"] = unified_state.user_responses
    state["user_interaction_required"] = unified_state.user_interaction_required
    
    return state


def route_after_analysis(state: State) -> Literal["validate_analysis", "tools", "handle_user_input", "__end__"]:
    """Enhanced routing with cycle detection."""
    # Add transition to cycle detector
    cycle = _cycle_detector.add_transition("analyze_logs", "route", state)
    
    if cycle:
        logger.warning(f"Breaking {cycle.cycle_type} cycle: {' -> '.join(cycle.pattern)}")
        return "__end__"
    
    # Check for user interaction
    if state.get("user_interaction_required") and state.get("pending_questions"):
        return "handle_user_input"
    
    # Check for tool calls
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        return "tools"
    
    # Default to validation
    return "validate_analysis"


def route_after_validation(state: State) -> Literal["analyze_logs", "__end__"]:
    """Enhanced routing after validation."""
    cycle = _cycle_detector.add_transition("validate_analysis", "route", state)
    
    if cycle and cycle.cycle_type in [CycleType.OSCILLATION, CycleType.DEADLOCK]:
        logger.warning(f"Breaking validation {cycle.cycle_type}")
        return "__end__"
    
    validation_status = state.get("validation_status", "")
    
    if validation_status == "valid":
        # Save final result if memory enabled
        if "memory" in state.get("enabled_features", []):
            asyncio.create_task(save_final_result(state))
        return "__end__"
    
    # Check retry limit
    node_visits = state.get("node_visits", {})
    if node_visits.get("analyze_logs", 0) >= 3:
        return "__end__"
    
    return "analyze_logs"


async def save_final_result(state: State):
    """Save final analysis result to memory."""
    if state.get("analysis_result") and state.get("checkpoint_id"):
        memory_manager = get_memory_manager()
        await memory_manager.save_analysis_result(
            state["checkpoint_id"],
            state["analysis_result"]
        )


def create_enhanced_graph(features: Optional[Set[str]] = None) -> StateGraph:
    """Create enhanced graph with all advanced features.
    
    Features:
        - interactive: User interaction support
        - memory: Persistence and history
        - streaming: Large log streaming
        - specialized: Domain-specific analyzers
        - caching: Result caching
        - monitoring: Resource tracking
    """
    features = features or {"caching", "specialized", "monitoring"}
    
    # Create workflow
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("analyze_logs", enhanced_analyze_logs)
    workflow.add_node("validate_analysis", validate_analysis)
    
    # Create enhanced tool node
    tools = [search_documentation, extract_patterns, generate_diagnostic_commands]
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)
    
    # Add interactive node if enabled
    if "interactive" in features:
        workflow.add_node("handle_user_input", handle_user_input)
    
    # Define edges
    workflow.add_edge(START, "analyze_logs")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "analyze_logs",
        route_after_analysis,
        {
            "validate_analysis": "validate_analysis",
            "tools": "tools",
            "handle_user_input": "handle_user_input" if "interactive" in features else END,
            "__end__": END
        }
    )
    
    workflow.add_conditional_edges(
        "validate_analysis",
        route_after_validation,
        {
            "analyze_logs": "analyze_logs",
            "__end__": END
        }
    )
    
    workflow.add_edge("tools", "validate_analysis")
    
    if "interactive" in features:
        workflow.add_edge("handle_user_input", "analyze_logs")
    
    # Add checkpointer if memory enabled
    checkpointer = None
    if "memory" in features:
        # Use async SQLite checkpointer
        checkpointer = AsyncSqliteSaver.from_conn_string("log_analyzer_checkpoints.db")
    
    # Compile graph
    compiled = workflow.compile(checkpointer=checkpointer)
    
    # Start resource monitoring if enabled
    if "monitoring" in features:
        resource_tracker = get_resource_tracker()
        asyncio.create_task(resource_tracker.start_monitoring())
    
    return compiled


# Export the enhanced graph
enhanced_graph = create_enhanced_graph()

# Also export a fully-featured graph
full_featured_graph = create_enhanced_graph({
    "interactive", "memory", "streaming", "specialized", "caching", "monitoring"
})