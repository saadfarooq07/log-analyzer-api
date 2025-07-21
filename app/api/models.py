"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid


class LogAnalysisRequest(BaseModel):
    """Request model for log analysis."""
    
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
    """Response model for log analysis."""
    
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique analysis ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    status: str = Field("completed", description="Analysis status")
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