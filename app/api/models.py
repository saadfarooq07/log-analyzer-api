"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Set
from datetime import datetime
import uuid


class LogAnalysisRequest(BaseModel):
    """Request model for log analysis with full feature support."""
    
    log_content: str = Field(
        ...,
        description="Log content to analyze",
        min_length=1,
        max_length=10_000_000  # 10MB limit
    )
    environment_details: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Environment context (OS, service versions, etc.)"
    )
    application_name: Optional[str] = Field(
        None,
        description="Name of the application that generated the logs"
    )
    analysis_type: Optional[str] = Field(
        "general",
        description="Type of analysis: general, security, performance, error"
    )
    include_suggestions: bool = Field(
        True,
        description="Include fix suggestions in the response"
    )
    include_documentation: bool = Field(
        True,
        description="Include documentation references"
    )
    
    # Advanced feature support
    user_id: Optional[str] = Field(
        None,
        description="User ID for session tracking and personalization"
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for maintaining context across requests"
    )
    requested_features: List[str] = Field(
        default_factory=list,
        description="List of features to enable: interactive, streaming, memory, caching, specialized, monitoring"
    )
    
    # Interactive features
    enable_interactive: bool = Field(
        False,
        description="Enable interactive mode for clarification questions"
    )
    
    # Streaming features
    enable_streaming: bool = Field(
        False,
        description="Enable streaming for large log files (auto-enabled for >10MB)"
    )
    chunk_size: Optional[int] = Field(
        None,
        description="Chunk size for streaming processing (bytes)"
    )
    
    # Memory features
    enable_memory: bool = Field(
        False,
        description="Enable memory features for context retention"
    )
    use_application_context: bool = Field(
        True,
        description="Use application-specific context from memory"
    )
    
    # Performance features
    enable_caching: bool = Field(
        True,
        description="Enable result caching for performance"
    )
    enable_specialized: bool = Field(
        True,
        description="Enable specialized analyzers for specific log types"
    )
    
    # Monitoring features
    enable_monitoring: bool = Field(
        False,
        description="Enable resource monitoring and metrics collection"
    )
    
    # Analysis configuration
    max_iterations: Optional[int] = Field(
        None,
        description="Maximum analysis iterations (default: 10)"
    )
    confidence_threshold: Optional[float] = Field(
        None,
        description="Minimum confidence threshold for results (0.0-1.0)"
    )
    
    def get_enabled_features(self) -> Set[str]:
        """Get set of enabled features based on request parameters."""
        features = set(self.requested_features)
        
        if self.enable_interactive:
            features.add("interactive")
        if self.enable_streaming or len(self.log_content) > 10 * 1024 * 1024:
            features.add("streaming")
        if self.enable_memory:
            features.add("memory")
        if self.enable_caching:
            features.add("caching")
        if self.enable_specialized:
            features.add("specialized")
        if self.enable_monitoring:
            features.add("monitoring")
        
        return features


class Issue(BaseModel):
    """Model for identified issues."""
    
    type: str = Field(..., description="Type of issue (error, warning, security, performance)")
    description: str = Field(..., description="Detailed description of the issue")
    severity: str = Field(..., description="Severity level: critical, high, medium, low")
    line_number: Optional[int] = Field(None, description="Line number where issue was found")
    timestamp: Optional[str] = Field(None, description="Timestamp of the issue if available")
    pattern: Optional[str] = Field(None, description="Pattern that matched this issue")


class Suggestion(BaseModel):
    """Model for fix suggestions."""
    
    issue_type: str = Field(..., description="Type of issue this suggestion addresses")
    suggestion: str = Field(..., description="Detailed suggestion for fixing the issue")
    priority: str = Field(..., description="Priority: immediate, high, medium, low")
    estimated_impact: Optional[str] = Field(None, description="Expected impact of implementing this fix")


class DocumentationReference(BaseModel):
    """Model for documentation references."""
    
    title: str = Field(..., description="Title of the documentation")
    url: str = Field(..., description="URL to the documentation")
    relevance: str = Field(..., description="How relevant this is to the issues found")
    excerpt: Optional[str] = Field(None, description="Relevant excerpt from the documentation")


class DiagnosticCommand(BaseModel):
    """Model for diagnostic commands."""
    
    command: str = Field(..., description="The diagnostic command to run")
    description: str = Field(..., description="What this command checks")
    platform: Optional[str] = Field(None, description="Platform where this command works (linux, windows, etc.)")


class AnalysisMetrics(BaseModel):
    """Performance metrics for the analysis."""
    
    total_lines: int = Field(..., description="Total number of log lines analyzed")
    issues_found: int = Field(..., description="Total number of issues found")
    processing_time: float = Field(..., description="Time taken to analyze in seconds")
    log_size_mb: float = Field(..., description="Size of the log file in MB")


class LogAnalysisResponse(BaseModel):
    """Response model for log analysis with full feature support."""
    
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique analysis ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    status: str = Field("completed", description="Analysis status: completed, pending, error, interactive_required")
    issues: List[Issue] = Field(default_factory=list, description="List of identified issues")
    suggestions: List[Suggestion] = Field(default_factory=list, description="List of fix suggestions")
    documentation_references: List[DocumentationReference] = Field(
        default_factory=list,
        description="Relevant documentation links"
    )
    diagnostic_commands: List[DiagnosticCommand] = Field(
        default_factory=list,
        description="Commands to run for further diagnosis"
    )
    summary: Optional[str] = Field(None, description="Executive summary of the analysis")
    metrics: Optional[AnalysisMetrics] = Field(None, description="Performance metrics")
    
    # Advanced feature results
    confidence_score: Optional[float] = Field(None, description="Overall confidence in analysis (0.0-1.0)")
    root_cause: Optional[str] = Field(None, description="Identified root cause of issues")
    
    # Interactive features
    pending_questions: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Questions requiring user input for interactive mode"
    )
    interaction_required: bool = Field(False, description="Whether user interaction is required")
    
    # Memory features
    memory_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Context retrieved from memory for this analysis"
    )
    similar_issues: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Similar issues found in memory"
    )
    
    # Streaming features
    chunk_summaries: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Summaries of processed chunks for streaming analysis"
    )
    total_chunks: Optional[int] = Field(None, description="Total number of chunks processed")
    
    # Specialized analysis results
    specialized_findings: Optional[Dict[str, Any]] = Field(
        None,
        description="Results from specialized analyzers (HDFS, security, etc.)"
    )
    
    # Execution metadata
    execution_summary: Optional[Dict[str, Any]] = Field(
        None,
        description="Summary of execution including node visits, tool calls, etc."
    )
    features_used: List[str] = Field(
        default_factory=list,
        description="List of features that were enabled for this analysis"
    )
    
    # Performance and monitoring
    resource_usage: Optional[Dict[str, Any]] = Field(
        None,
        description="Resource usage metrics if monitoring is enabled"
    )
    cache_hit: Optional[bool] = Field(None, description="Whether result was retrieved from cache")
    
    # Error handling
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings during analysis")
    cycle_detection: Optional[Dict[str, Any]] = Field(
        None,
        description="Cycle detection information if cycles were detected"
    )


class StreamEvent(BaseModel):
    """Model for streaming events."""
    
    type: str = Field(..., description="Event type: progress, result, error")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Model for error responses."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)