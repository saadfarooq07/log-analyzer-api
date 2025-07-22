"""API routes for log analysis."""

from fastapi import APIRouter, HTTPException, Response, BackgroundTasks
from typing import AsyncGenerator, Dict, Any, List, Optional
import json
import asyncio
import time
import logging
import uuid
from datetime import datetime, timedelta
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
from ..agent.graph import create_graph, ENHANCED_FEATURES_AVAILABLE, get_performance_metrics
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
        
        # Prepare input state with full feature support
        enabled_features = request.get_enabled_features()
        
        input_state = {
            "log_content": request.log_content,
            "environment_details": request.environment_details or {},
            "application_name": request.application_name,
            "analysis_type": request.analysis_type,
            "include_suggestions": request.include_suggestions,
            "include_documentation": request.include_documentation,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "enabled_features": list(enabled_features),
            "user_interaction_required": False,
            "interaction_history": [],
            "is_streaming": "streaming" in enabled_features,
            "current_chunk_index": 0,
            "total_chunks": 1,
            "chunk_results": [],
            "checkpoint_metadata": {},
            "memory_matches": None,
            "application_context": None,
            "user_context": None,
            "save_count": 0
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
        
        # Build comprehensive response
        response = LogAnalysisResponse(
            issues=issues,
            suggestions=suggestions,
            documentation_references=doc_refs,
            diagnostic_commands=diag_cmds,
            summary=analysis_result.get("summary"),
            metrics=metrics,
            confidence_score=analysis_result.get("confidence_score"),
            root_cause=analysis_result.get("root_cause"),
            pending_questions=result.get("pending_questions"),
            interaction_required=result.get("user_interaction_required", False),
            memory_context=result.get("memory_context"),
            similar_issues=result.get("similar_issues"),
            chunk_summaries=result.get("chunk_results"),
            total_chunks=result.get("total_chunks"),
            specialized_findings=analysis_result.get("specialized_findings"),
            execution_summary=result.get("node_visits"),
            features_used=list(enabled_features),
            resource_usage=analysis_result.get("resource_usage"),
            cache_hit=False,  # Will be set by caching layer
            warnings=analysis_result.get("warnings", []),
            cycle_detection=analysis_result.get("cycle_detection")
        )
        
        return response
        
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
            
            # Prepare input state with full feature support
            enabled_features = request.get_enabled_features()
            
            input_state = {
                "log_content": request.log_content,
                "environment_details": request.environment_details or {},
                "application_name": request.application_name,
                "analysis_type": request.analysis_type,
                "include_suggestions": request.include_suggestions,
                "include_documentation": request.include_documentation,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "enabled_features": list(enabled_features),
                "user_interaction_required": False,
                "interaction_history": [],
                "is_streaming": True,  # Always true for streaming endpoint
                "current_chunk_index": 0,
                "total_chunks": 1,
                "chunk_results": [],
                "checkpoint_metadata": {},
                "memory_matches": None,
                "application_context": None,
                "user_context": None,
                "save_count": 0
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
        start_time = time.time()
        
        try:
            # Force enable interactive features
            enabled_features = request.get_enabled_features()
            enabled_features.add("interactive")
            enabled_features.add("caching")
            enabled_features.add("specialized")
            
            # Check log size
            log_size_mb = len(request.log_content.encode()) / (1024 * 1024)
            if log_size_mb > settings.max_log_size_mb:
                raise HTTPException(
                    status_code=413,
                    detail=f"Log size {log_size_mb:.2f}MB exceeds maximum allowed size of {settings.max_log_size_mb}MB"
                )
            
            # Create analysis ID for tracking
            analysis_id = str(uuid.uuid4())
            
            # Prepare input state with interactive features
            input_state = {
                "log_content": request.log_content,
                "environment_details": request.environment_details or {},
                "application_name": request.application_name,
                "analysis_type": request.analysis_type,
                "include_suggestions": request.include_suggestions,
                "include_documentation": request.include_documentation,
                "user_id": request.user_id,
                "session_id": request.session_id or analysis_id,
                "enabled_features": list(enabled_features),
                "user_interaction_required": False,
                "interaction_history": [],
                "is_streaming": "streaming" in enabled_features,
                "current_chunk_index": 0,
                "total_chunks": 1,
                "chunk_results": [],
                "checkpoint_metadata": {},
                "memory_matches": None,
                "application_context": None,
                "user_context": None,
                "save_count": 0
            }
            
            # Create graph with interactive features
            graph = create_graph(enabled_features)
            
            # Run analysis
            result = await asyncio.wait_for(
                graph.ainvoke(input_state),
                timeout=settings.analysis_timeout
            )
            
            # Extract analysis result
            analysis_result = result.get("analysis_result", {})
            
            # Check for pending questions
            response_data = {
                "analysis_id": analysis_id,
                "timestamp": datetime.utcnow(),
                "status": "pending" if result.get("pending_questions") else "completed",
                "features_used": list(enabled_features),
                "processing_time": time.time() - start_time
            }
            
            if result.get("pending_questions"):
                response_data["pending_questions"] = result["pending_questions"]
                response_data["interaction_required"] = True
                response_data["partial_analysis"] = analysis_result
            else:
                # Process complete results
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
                
                response_data["analysis_result"] = {
                    "issues": [issue.dict() for issue in issues],
                    "summary": analysis_result.get("summary"),
                    "confidence_score": analysis_result.get("confidence_score"),
                    "root_cause": analysis_result.get("root_cause"),
                    "specialized_findings": analysis_result.get("specialized_findings")
                }
            
            return response_data
            
        except asyncio.TimeoutError:
            logger.error(f"Interactive analysis timeout after {settings.analysis_timeout}s")
            raise HTTPException(
                status_code=504,
                detail=f"Analysis timeout after {settings.analysis_timeout} seconds"
            )
        except Exception as e:
            logger.error(f"Interactive analysis error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @router.post("/analyze/continue/{analysis_id}")
    async def continue_interactive_analysis(
        analysis_id: str,
        answers: Dict[str, Any]
    ):
        """Continue interactive analysis with user answers."""
        try:
            # For now, simulate continuation logic
            # In a full implementation, this would:
            # 1. Retrieve the saved state from memory/database
            # 2. Apply the user answers to the state
            # 3. Continue the graph execution
            # 4. Return updated results
            
            logger.info(f"Continuing interactive analysis {analysis_id} with answers: {answers}")
            
            # Simulate processing the answers
            await asyncio.sleep(1)  # Simulate processing time
            
            return {
                "analysis_id": analysis_id,
                "status": "completed",
                "timestamp": datetime.utcnow(),
                "message": "Analysis continued with user input",
                "answers_processed": answers,
                "final_result": {
                    "summary": "Analysis completed with user clarifications",
                    "confidence_score": 0.95,
                    "additional_insights": "User input helped clarify ambiguous log patterns"
                }
            }
            
        except Exception as e:
            logger.error(f"Error continuing interactive analysis: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
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
        application_name: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """Get analysis history."""
        try:
            # Simulate history retrieval
            # In a full implementation, this would query a database
            
            history_items = []
            
            # Generate sample history for demonstration
            for i in range(min(limit, 5)):
                history_items.append({
                    "analysis_id": f"hist_{uuid.uuid4().hex[:8]}",
                    "timestamp": (datetime.utcnow() - timedelta(days=i)).isoformat(),
                    "application_name": application_name or f"app_{i}",
                    "user_id": user_id or "demo_user",
                    "status": "completed",
                    "issues_found": 3 - i,
                    "confidence_score": 0.8 + (i * 0.05),
                    "summary": f"Analysis {i+1}: Found {3-i} issues in application logs"
                })
            
            return {
                "history": history_items,
                "total": len(history_items),
                "limit": limit,
                "filters": {
                    "application_name": application_name,
                    "user_id": user_id
                },
                "message": "Sample history data - full implementation requires database"
            }
            
        except Exception as e:
            logger.error(f"Error retrieving history: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @router.get("/patterns/{pattern_type}")
    async def get_known_patterns(pattern_type: str):
        """Get known patterns of a specific type."""
        try:
            # Simulate pattern retrieval
            # In a full implementation, this would query a pattern database
            
            pattern_templates = {
                "error": [
                    {
                        "pattern": "Database connection timeout",
                        "frequency": 45,
                        "severity": "high",
                        "common_causes": ["Network issues", "Database overload", "Connection pool exhaustion"],
                        "suggested_fixes": ["Check network connectivity", "Increase connection timeout", "Scale database"]
                    },
                    {
                        "pattern": "Out of memory error",
                        "frequency": 23,
                        "severity": "critical",
                        "common_causes": ["Memory leak", "Insufficient heap size", "Large object allocation"],
                        "suggested_fixes": ["Increase heap size", "Profile memory usage", "Optimize object lifecycle"]
                    }
                ],
                "security": [
                    {
                        "pattern": "Failed login attempts",
                        "frequency": 156,
                        "severity": "medium",
                        "common_causes": ["Brute force attack", "User credential issues", "Account lockout"],
                        "suggested_fixes": ["Implement rate limiting", "Enable account lockout", "Monitor IP patterns"]
                    }
                ],
                "performance": [
                    {
                        "pattern": "Slow query execution",
                        "frequency": 78,
                        "severity": "medium",
                        "common_causes": ["Missing indexes", "Large dataset", "Complex joins"],
                        "suggested_fixes": ["Add database indexes", "Optimize queries", "Implement caching"]
                    }
                ]
            }
            
            patterns = pattern_templates.get(pattern_type.lower(), [])
            
            return {
                "pattern_type": pattern_type,
                "patterns": patterns,
                "total_patterns": len(patterns),
                "message": f"Sample {pattern_type} patterns - full implementation requires pattern database"
            }
            
        except Exception as e:
            logger.error(f"Error retrieving patterns: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @router.post("/analyze/advanced")
    async def analyze_logs_advanced(request: LogAnalysisRequest):
        """Perform advanced analysis with all features enabled."""
        start_time = time.time()
        
        try:
            # Enable all advanced features
            enabled_features = {
                "interactive", "caching", "specialized", "monitoring", "streaming"
            }
            
            # Add memory if requested
            if request.enable_memory:
                enabled_features.add("memory")
            
            # Check log size
            log_size_mb = len(request.log_content.encode()) / (1024 * 1024)
            if log_size_mb > settings.max_log_size_mb:
                raise HTTPException(
                    status_code=413,
                    detail=f"Log size {log_size_mb:.2f}MB exceeds maximum allowed size of {settings.max_log_size_mb}MB"
                )
            
            logger.info(f"Starting advanced analysis for {log_size_mb:.2f}MB log file with features: {enabled_features}")
            
            # Prepare comprehensive input state
            input_state = {
                "log_content": request.log_content,
                "environment_details": request.environment_details or {},
                "application_name": request.application_name,
                "analysis_type": request.analysis_type,
                "include_suggestions": request.include_suggestions,
                "include_documentation": request.include_documentation,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "enabled_features": list(enabled_features),
                "user_interaction_required": False,
                "interaction_history": [],
                "is_streaming": log_size_mb > 1.0,  # Auto-enable streaming for >1MB
                "current_chunk_index": 0,
                "total_chunks": max(1, int(log_size_mb)),
                "chunk_results": [],
                "checkpoint_metadata": {},
                "memory_matches": None,
                "application_context": None,
                "user_context": None,
                "save_count": 0
            }
            
            # Create advanced graph
            graph = create_graph(enabled_features)
            
            # Run analysis with timeout
            result = await asyncio.wait_for(
                graph.ainvoke(input_state),
                timeout=settings.analysis_timeout * 2  # Double timeout for advanced analysis
            )
            
            # Extract comprehensive results
            analysis_result = result.get("analysis_result", {})
            
            # Process all result types
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
            
            suggestions = []
            for sug_data in analysis_result.get("suggestions", []):
                suggestions.append(Suggestion(
                    issue_type=sug_data.get("issue_type", "general"),
                    suggestion=sug_data.get("suggestion", ""),
                    priority=sug_data.get("priority", "medium"),
                    estimated_impact=sug_data.get("estimated_impact")
                ))
            
            doc_refs = []
            for ref_data in analysis_result.get("documentation_references", []):
                doc_refs.append(DocumentationReference(
                    title=ref_data.get("title", "Documentation"),
                    url=ref_data.get("url", ""),
                    relevance=ref_data.get("relevance", "related"),
                    excerpt=ref_data.get("excerpt")
                ))
            
            diag_cmds = []
            for cmd_data in analysis_result.get("diagnostic_commands", []):
                diag_cmds.append(DiagnosticCommand(
                    command=cmd_data.get("command", ""),
                    description=cmd_data.get("description", ""),
                    platform=cmd_data.get("platform")
                ))
            
            # Calculate comprehensive metrics
            processing_time = time.time() - start_time
            total_lines = len(request.log_content.splitlines())
            
            metrics = AnalysisMetrics(
                total_lines=total_lines,
                issues_found=len(issues),
                processing_time=processing_time,
                log_size_mb=log_size_mb
            )
            
            # Get performance metrics
            perf_metrics = get_performance_metrics()
            
            logger.info(f"Advanced analysis completed in {processing_time:.2f}s, found {len(issues)} issues")
            
            # Build comprehensive response
            response = LogAnalysisResponse(
                issues=issues,
                suggestions=suggestions,
                documentation_references=doc_refs,
                diagnostic_commands=diag_cmds,
                summary=analysis_result.get("summary", "Advanced analysis completed"),
                metrics=metrics,
                confidence_score=analysis_result.get("confidence_score", 0.85),
                root_cause=analysis_result.get("root_cause"),
                pending_questions=result.get("pending_questions"),
                interaction_required=result.get("user_interaction_required", False),
                memory_context=result.get("memory_context"),
                similar_issues=result.get("similar_issues"),
                chunk_summaries=result.get("chunk_results"),
                total_chunks=result.get("total_chunks"),
                specialized_findings=analysis_result.get("specialized_findings"),
                execution_summary={
                    **result.get("node_visits", {}),
                    "performance_metrics": perf_metrics
                },
                features_used=list(enabled_features),
                resource_usage=analysis_result.get("resource_usage"),
                cache_hit=False,
                warnings=analysis_result.get("warnings", []),
                cycle_detection=perf_metrics.get("cycle_summary")
            )
            
            return response
            
        except asyncio.TimeoutError:
            logger.error(f"Advanced analysis timeout after {settings.analysis_timeout * 2}s")
            raise HTTPException(
                status_code=504,
                detail=f"Advanced analysis timeout after {settings.analysis_timeout * 2} seconds"
            )
        except Exception as e:
            logger.error(f"Advanced analysis error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @router.get("/features")
    async def get_available_features():
        """Get information about available features and their status."""
        return {
            "available_features": {
                "interactive": {
                    "description": "Interactive mode with clarification questions",
                    "status": "available",
                    "requirements": []
                },
                "streaming": {
                    "description": "Streaming analysis for large log files",
                    "status": "available",
                    "requirements": []
                },
                "caching": {
                    "description": "Result caching for improved performance",
                    "status": "available",
                    "requirements": []
                },
                "specialized": {
                    "description": "Specialized analyzers for specific log types",
                    "status": "available",
                    "requirements": []
                },
                "monitoring": {
                    "description": "Resource monitoring and performance tracking",
                    "status": "available",
                    "requirements": []
                },
                "memory": {
                    "description": "Memory and context retention across sessions",
                    "status": "limited",
                    "requirements": ["Database configuration"]
                }
            },
            "enhanced_features_available": ENHANCED_FEATURES_AVAILABLE,
            "performance_metrics": get_performance_metrics(),
            "configuration": {
                "max_log_size_mb": settings.max_log_size_mb,
                "analysis_timeout": settings.analysis_timeout,
                "streaming_enabled": settings.enable_streaming
            }
        }