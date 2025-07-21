"""API routes for log analysis."""

from fastapi import APIRouter, HTTPException, Response, BackgroundTasks
from typing import AsyncGenerator, Dict, Any, List, Optional
import json
import asyncio
import time
import logging
import uuid
from sse_starlette.sse import EventSourceResponse

from .models import (
    LogAnalysisRequest,
    LogAnalysisResponse,
    Issue,
    Suggestion,
    DocumentationReference,
    DiagnosticCommand,
    AnalysisMetrics,
    StreamEvent,
    ErrorResponse
)
from ..agent.graph import create_graph, ENHANCED_FEATURES_AVAILABLE
from ..config import settings

# Import enhanced features if available
if ENHANCED_FEATURES_AVAILABLE:
    from ..agent.cache_manager import get_cache_manager
    from ..agent.resource_tracker import get_resource_tracker
    from ..agent.circuit_breaker import get_circuit_breaker_manager
    from ..agent.interactive_handler import get_interactive_handler
    from ..agent.memory_manager import get_memory_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=LogAnalysisResponse)
async def analyze_logs(request: LogAnalysisRequest):
    """Analyze logs synchronously and return complete results."""
    start_time = time.time()
    
    try:
        # Check log size
        log_size_mb = len(request.log_content.encode()) / (1024 * 1024)
        if log_size_mb > settings.max_log_size_mb:
            raise HTTPException(
                status_code=413,
                detail=f"Log size {log_size_mb:.2f}MB exceeds maximum allowed size of {settings.max_log_size_mb}MB"
            )
        
        logger.info(f"Starting analysis for {log_size_mb:.2f}MB log file")
        
        # Create graph
        graph = create_graph()
        
        # Prepare input state
        input_state = {
            "log_content": request.log_content,
            "environment_details": request.environment_details or {},
            "application_name": request.application_name,
            "analysis_type": request.analysis_type,
            "include_suggestions": request.include_suggestions,
            "include_documentation": request.include_documentation
        }
        
        # Run analysis
        result = await asyncio.wait_for(
            graph.ainvoke(input_state),
            timeout=settings.analysis_timeout
        )
        
        # Extract analysis result
        analysis_result = result.get("analysis_result", {})
        
        # Process issues
        issues = []
        for issue_data in analysis_result.get("issues", []):
            issues.append(Issue(
                type=issue_data.get("type", "unknown"),
                description=issue_data.get("description", ""),
                severity=issue_data.get("severity", "medium"),
                line_number=issue_data.get("line_number"),
                timestamp=issue_data.get("timestamp"),
                pattern=issue_data.get("pattern")
            ))
        
        # Process suggestions
        suggestions = []
        if request.include_suggestions:
            for sug_data in analysis_result.get("suggestions", []):
                suggestions.append(Suggestion(
                    issue_type=sug_data.get("issue_type", "general"),
                    suggestion=sug_data.get("suggestion", ""),
                    priority=sug_data.get("priority", "medium"),
                    estimated_impact=sug_data.get("estimated_impact")
                ))
        
        # Process documentation references
        doc_refs = []
        if request.include_documentation:
            for ref_data in analysis_result.get("documentation_references", []):
                doc_refs.append(DocumentationReference(
                    title=ref_data.get("title", "Documentation"),
                    url=ref_data.get("url", ""),
                    relevance=ref_data.get("relevance", "related"),
                    excerpt=ref_data.get("excerpt")
                ))
        
        # Process diagnostic commands
        diag_cmds = []
        for cmd_data in analysis_result.get("diagnostic_commands", []):
            diag_cmds.append(DiagnosticCommand(
                command=cmd_data.get("command", ""),
                description=cmd_data.get("description", ""),
                platform=cmd_data.get("platform")
            ))
        
        # Calculate metrics
        processing_time = time.time() - start_time
        total_lines = len(request.log_content.splitlines())
        
        metrics = AnalysisMetrics(
            total_lines=total_lines,
            issues_found=len(issues),
            processing_time=processing_time,
            log_size_mb=log_size_mb
        )
        
        logger.info(f"Analysis completed in {processing_time:.2f}s, found {len(issues)} issues")
        
        return LogAnalysisResponse(
            issues=issues,
            suggestions=suggestions,
            documentation_references=doc_refs,
            diagnostic_commands=diag_cmds,
            summary=analysis_result.get("summary"),
            metrics=metrics
        )
        
    except asyncio.TimeoutError:
        logger.error(f"Analysis timeout after {settings.analysis_timeout}s")
        raise HTTPException(
            status_code=504,
            detail=f"Analysis timeout after {settings.analysis_timeout} seconds"
        )
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/stream")
async def analyze_logs_stream(request: LogAnalysisRequest):
    """Analyze logs with Server-Sent Events (SSE) streaming."""
    
    async def event_generator() -> AsyncGenerator[str, None]:
        start_time = time.time()
        
        try:
            # Check log size
            log_size_mb = len(request.log_content.encode()) / (1024 * 1024)
            if log_size_mb > settings.max_log_size_mb:
                yield json.dumps({
                    "type": "error",
                    "data": {
                        "error": "Log size exceeds limit",
                        "message": f"Log size {log_size_mb:.2f}MB exceeds maximum allowed size of {settings.max_log_size_mb}MB"
                    }
                })
                return
            
            # Send initial progress
            yield json.dumps({
                "type": "progress",
                "data": {
                    "status": "starting",
                    "message": "Initializing analysis...",
                    "log_size_mb": log_size_mb
                }
            })
            
            # Create graph
            graph = create_graph()
            
            # Prepare input state
            input_state = {
                "log_content": request.log_content,
                "environment_details": request.environment_details or {},
                "application_name": request.application_name,
                "analysis_type": request.analysis_type,
                "include_suggestions": request.include_suggestions,
                "include_documentation": request.include_documentation
            }
            
            # Stream analysis events
            event_count = 0
            async for event in graph.astream(input_state):
                event_count += 1
                
                # Send progress updates
                if "analyzing" in str(event).lower():
                    yield json.dumps({
                        "type": "progress",
                        "data": {
                            "status": "analyzing",
                            "message": "Analyzing log patterns...",
                            "progress": min(event_count * 10, 90)
                        }
                    })
                
                # Check for intermediate results
                if "issues" in event:
                    yield json.dumps({
                        "type": "partial_result",
                        "data": {
                            "issues_found": len(event.get("issues", [])),
                            "message": f"Found {len(event.get('issues', []))} issues so far..."
                        }
                    })
                
                await asyncio.sleep(0.1)  # Small delay for SSE
            
            # Get final result
            final_result = await graph.ainvoke(input_state)
            analysis_result = final_result.get("analysis_result", {})
            
            # Send final result
            processing_time = time.time() - start_time
            yield json.dumps({
                "type": "complete",
                "data": {
                    "analysis_id": str(analysis_result.get("analysis_id", "")),
                    "issues": analysis_result.get("issues", []),
                    "suggestions": analysis_result.get("suggestions", []) if request.include_suggestions else [],
                    "documentation_references": analysis_result.get("documentation_references", []) if request.include_documentation else [],
                    "diagnostic_commands": analysis_result.get("diagnostic_commands", []),
                    "summary": analysis_result.get("summary"),
                    "metrics": {
                        "total_lines": len(request.log_content.splitlines()),
                        "issues_found": len(analysis_result.get("issues", [])),
                        "processing_time": processing_time,
                        "log_size_mb": log_size_mb
                    }
                }
            })
            
        except Exception as e:
            logger.error(f"Streaming analysis error: {str(e)}")
            yield json.dumps({
                "type": "error",
                "data": {
                    "error": type(e).__name__,
                    "message": str(e)
                }
            })
    
    if not settings.enable_streaming:
        raise HTTPException(
            status_code=501,
            detail="Streaming is not enabled. Use /analyze endpoint instead."
        )
    
    return EventSourceResponse(event_generator())


@router.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get analysis results by ID (placeholder for future implementation)."""
    # This would typically fetch from a database or cache
    raise HTTPException(
        status_code=501,
        detail="Analysis retrieval not yet implemented. Use synchronous analysis for now."
    )


@router.post("/analyze/batch")
async def analyze_logs_batch(requests: List[LogAnalysisRequest], background_tasks: BackgroundTasks):
    """Analyze multiple logs in batch (placeholder for future implementation)."""
    raise HTTPException(
        status_code=501,
        detail="Batch analysis not yet implemented. Please analyze logs individually."
    )


# Enhanced feature endpoints (only available when enhanced features are enabled)
if ENHANCED_FEATURES_AVAILABLE:
    
    @router.post("/analyze/interactive")
    async def analyze_logs_interactive(request: LogAnalysisRequest):
        """Analyze logs with interactive mode enabled."""
        # Add interactive feature to the request
        input_state = {
            "log_content": request.log_content,
            "environment_details": request.environment_details or {},
            "application_name": request.application_name,
            "analysis_type": request.analysis_type,
            "enabled_features": ["interactive", "caching", "specialized"]
        }
        
        # Create analysis ID for tracking
        analysis_id = str(uuid.uuid4())
        input_state["analysis_id"] = analysis_id
        
        # Run initial analysis
        graph = create_graph()
        result = await graph.ainvoke(input_state)
        
        # Check for pending questions
        response_data = {
            "analysis_id": analysis_id,
            "status": "pending" if result.get("pending_questions") else "completed"
        }
        
        if result.get("pending_questions"):
            response_data["pending_questions"] = result["pending_questions"]
            response_data["interaction_required"] = True
        else:
            response_data["analysis_result"] = result.get("analysis_result", {})
        
        return response_data
    
    
    @router.post("/analyze/continue")
    async def continue_interactive_analysis(
        analysis_id: str,
        answers: Dict[str, Any]
    ):
        """Continue interactive analysis with user answers."""
        # Get interactive handler
        handler = get_interactive_handler()
        
        # TODO: Retrieve state from memory/cache
        # For now, return placeholder
        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "message": "Interactive continuation will be implemented with state persistence"
        }
    
    
    @router.get("/metrics/cache")
    async def get_cache_metrics():
        """Get cache performance metrics."""
        cache_manager = get_cache_manager()
        return cache_manager.get_performance_stats()
    
    
    @router.post("/cache/clear")
    async def clear_cache():
        """Clear all caches."""
        cache_manager = get_cache_manager()
        cache_manager.clear_all()
        return {"status": "success", "message": "All caches cleared"}
    
    
    @router.get("/metrics/resources")
    async def get_resource_metrics():
        """Get current resource usage metrics."""
        tracker = get_resource_tracker()
        return tracker.get_summary()
    
    
    @router.get("/metrics/circuit-breakers")
    async def get_circuit_breaker_status():
        """Get circuit breaker status."""
        manager = get_circuit_breaker_manager()
        return manager.get_all_stats()
    
    
    @router.get("/history")
    async def get_analysis_history(
        limit: int = 10,
        application_name: Optional[str] = None
    ):
        """Get analysis history."""
        memory_manager = get_memory_manager()
        await memory_manager.initialize()
        
        # TODO: Implement history retrieval
        return {
            "history": [],
            "total": 0,
            "message": "History retrieval will be implemented with memory persistence"
        }
    
    
    @router.get("/patterns/{pattern_type}")
    async def get_known_patterns(pattern_type: str):
        """Get known patterns of a specific type."""
        memory_manager = get_memory_manager()
        await memory_manager.initialize()
        
        # TODO: Implement pattern retrieval
        return {
            "pattern_type": pattern_type,
            "patterns": [],
            "message": "Pattern retrieval will be implemented with memory persistence"
        }