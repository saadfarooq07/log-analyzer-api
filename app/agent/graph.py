"""LangGraph workflow for log analysis with enhanced features."""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import logging
import os

from app.agent.state import State
from app.agent.nodes import analyze_logs, validate_analysis
from app.agent.tools import search_documentation, extract_patterns, generate_diagnostic_commands

# Import enhanced graph if available
try:
    from app.agent.enhanced_graph import create_enhanced_graph, enhanced_graph, full_featured_graph
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False

logger = logging.getLogger(__name__)


def create_graph():
    """Create the log analysis workflow graph."""
    
    # Check if enhanced features are requested
    use_enhanced = os.getenv("USE_ENHANCED_FEATURES", "false").lower() == "true"
    
    if use_enhanced and ENHANCED_FEATURES_AVAILABLE:
        logger.info("Using enhanced graph with advanced features")
        # Determine which features to enable
        features = set()
        
        if os.getenv("ENABLE_CACHING", "true").lower() == "true":
            features.add("caching")
        if os.getenv("ENABLE_SPECIALIZED", "true").lower() == "true":
            features.add("specialized")
        if os.getenv("ENABLE_MONITORING", "true").lower() == "true":
            features.add("monitoring")
        if os.getenv("ENABLE_INTERACTIVE", "false").lower() == "true":
            features.add("interactive")
        if os.getenv("ENABLE_MEMORY", "false").lower() == "true":
            features.add("memory")
        if os.getenv("ENABLE_STREAMING", "true").lower() == "true":
            features.add("streaming")
        
        return create_enhanced_graph(features)
    
    # Default basic graph
    logger.info("Using basic graph")
    
    # Create workflow
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("analyze", analyze_logs)
    workflow.add_node("validate", validate_analysis)
    
    # Create tool node with our tools
    tools = [search_documentation, extract_patterns, generate_diagnostic_commands]
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)
    
    # Define the flow
    workflow.set_entry_point("analyze")
    
    # Add edges
    workflow.add_edge("analyze", "validate")
    
    # Conditional edge from validate
    def should_continue(state: State) -> str:
        """Determine if we should continue or end."""
        validation_result = state.get("validation_result")
        
        # If validation passed or we have a result, end
        if (validation_result and validation_result.get("is_valid", False)) or state.get("analysis_result"):
            return "end"
        
        # If we have an error, end
        if state.get("error"):
            return "end"
        
        # Otherwise, we might want to retry or use tools
        # For now, we'll just end
        return "end"
    
    workflow.add_conditional_edges(
        "validate",
        should_continue,
        {
            "end": END,
            "tools": "tools"  # Future: could route to tools for enhancement
        }
    )
    
    # Tools always go back to validate
    workflow.add_edge("tools", "validate")
    
    # Compile the graph
    return workflow.compile()


def create_streaming_graph():
    """Create a graph optimized for streaming large logs."""
    if ENHANCED_FEATURES_AVAILABLE:
        # Use enhanced graph with streaming enabled
        return create_enhanced_graph({"streaming", "caching", "specialized"})
    
    # Fall back to basic graph
    return create_graph()


# Export the compiled graph for LangGraph Cloud
# Use enhanced graph by default if available and requested
if os.getenv("USE_ENHANCED_FEATURES", "false").lower() == "true" and ENHANCED_FEATURES_AVAILABLE:
    graph = enhanced_graph
else:
    graph = create_graph()