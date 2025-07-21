"""LangGraph workflow for log analysis."""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import logging

from .state import State
from .nodes import analyze_logs, validate_analysis
from .tools import search_documentation, extract_patterns, generate_diagnostic_commands

logger = logging.getLogger(__name__)


def create_graph():
    """Create the log analysis workflow graph."""
    
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
        validation_result = state.get("validation_result", {})
        
        # If validation passed or we have a result, end
        if validation_result.get("is_valid", False) or state.get("analysis_result"):
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
    # For now, use the same graph
    # In the future, this could be optimized for streaming
    return create_graph()


# Export the compiled graph for LangGraph Cloud
graph = create_graph()