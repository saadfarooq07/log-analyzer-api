"""State definitions for the log analyzer agent."""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage


class State(TypedDict):
    """Main state for the log analyzer workflow."""
    
    # Input fields
    log_content: str
    environment_details: Dict[str, Any]
    application_name: Optional[str]
    analysis_type: str
    include_suggestions: bool
    include_documentation: bool
    
    # Working fields
    messages: List[BaseMessage]
    current_analysis: Optional[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    
    # Output fields
    analysis_result: Optional[Dict[str, Any]]
    error: Optional[str]