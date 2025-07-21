"""Advanced streaming processor for handling large logs with parallel processing.

This module provides sophisticated streaming capabilities including parallel
chunk processing, memory-efficient analysis, and progressive results.
"""

import asyncio
import hashlib
from typing import Dict, Any, List, Optional, AsyncIterator, Tuple
from dataclasses import dataclass
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import aiofiles
from langchain_core.messages import HumanMessage, AIMessage

from app.agent.unified_state import UnifiedState
from app.agent.rate_limiter import get_rate_limiter
from app.agent.circuit_breaker import get_circuit_breaker

logger = logging.getLogger(__name__)


@dataclass
class LogChunk:
    """Represents a chunk of log data."""
    index: int
    content: str
    start_line: int
    end_line: int
    size_bytes: int
    hash: str
    
    @classmethod
    def create(cls, index: int, content: str, start_line: int, end_line: int) -> "LogChunk":
        """Create a log chunk with computed hash."""
        size_bytes = len(content.encode())
        chunk_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return cls(
            index=index,
            content=content,
            start_line=start_line,
            end_line=end_line,
            size_bytes=size_bytes,
            hash=chunk_hash
        )


class StreamingProcessor:
    """Advanced streaming processor for large log analysis."""
    
    def __init__(
        self,
        chunk_size_mb: float = 10.0,
        max_concurrent_chunks: int = 3,
        overlap_lines: int = 50
    ):
        """Initialize streaming processor.
        
        Args:
            chunk_size_mb: Target chunk size in MB
            max_concurrent_chunks: Maximum chunks to process in parallel
            overlap_lines: Number of lines to overlap between chunks
        """
        self.chunk_size_mb = chunk_size_mb
        self.chunk_size_bytes = int(chunk_size_mb * 1024 * 1024)
        self.max_concurrent_chunks = max_concurrent_chunks
        self.overlap_lines = overlap_lines
        
        # Processing state
        self.processed_chunks: Dict[int, Dict[str, Any]] = {}
        self.processing_semaphore = asyncio.Semaphore(max_concurrent_chunks)
        
        # Metrics
        self.total_chunks = 0
        self.processed_count = 0
        self.start_time = None
        self.end_time = None
    
    async def process_log_stream(
        self,
        log_content: str,
        state: UnifiedState,
        analyze_func: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process log content as a stream with parallel chunk processing.
        
        Args:
            log_content: Full log content
            state: Unified state object
            analyze_func: Function to analyze each chunk
            
        Yields:
            Progressive analysis results
        """
        self.start_time = time.time()
        
        # Create chunks
        chunks = self._create_chunks(log_content)
        self.total_chunks = len(chunks)
        
        logger.info(f"Created {self.total_chunks} chunks for streaming analysis")
        
        # Update state
        state.is_streaming = True
        state.total_chunks = self.total_chunks
        
        # Process chunks in parallel
        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(
                self._process_chunk_with_semaphore(chunk, state, analyze_func)
            )
            tasks.append(task)
        
        # Yield results as they complete
        for task in asyncio.as_completed(tasks):
            try:
                chunk_result = await task
                if chunk_result:
                    self.processed_count += 1
                    
                    # Update state
                    state.current_chunk_index = chunk_result["chunk_index"]
                    state.chunk_results.append(chunk_result)
                    
                    # Yield progress
                    yield {
                        "type": "chunk_complete",
                        "chunk_index": chunk_result["chunk_index"],
                        "total_chunks": self.total_chunks,
                        "progress": (self.processed_count / self.total_chunks) * 100,
                        "issues_found": len(chunk_result.get("issues", [])),
                        "chunk_result": chunk_result
                    }
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                yield {
                    "type": "chunk_error",
                    "error": str(e)
                }
        
        # Merge all results
        merged_result = self._merge_chunk_results(list(self.processed_chunks.values()))
        
        self.end_time = time.time()
        processing_time = self.end_time - self.start_time
        
        # Final result
        yield {
            "type": "stream_complete",
            "final_result": merged_result,
            "metrics": {
                "total_chunks": self.total_chunks,
                "processing_time": processing_time,
                "chunks_per_second": self.total_chunks / processing_time if processing_time > 0 else 0,
                "parallel_efficiency": (processing_time / self.total_chunks) if self.total_chunks > 0 else 0
            }
        }
    
    def _create_chunks(self, log_content: str) -> List[LogChunk]:
        """Create chunks from log content with overlap."""
        lines = log_content.splitlines()
        chunks = []
        
        current_chunk_lines = []
        current_size = 0
        start_line = 0
        chunk_index = 0
        
        for i, line in enumerate(lines):
            line_size = len(line.encode()) + 1  # +1 for newline
            
            # Check if adding this line would exceed chunk size
            if current_size + line_size > self.chunk_size_bytes and current_chunk_lines:
                # Create chunk
                chunk_content = "\n".join(current_chunk_lines)
                chunks.append(LogChunk.create(
                    index=chunk_index,
                    content=chunk_content,
                    start_line=start_line,
                    end_line=i - 1
                ))
                
                # Prepare next chunk with overlap
                overlap_start = max(0, len(current_chunk_lines) - self.overlap_lines)
                current_chunk_lines = current_chunk_lines[overlap_start:]
                current_size = sum(len(line.encode()) + 1 for line in current_chunk_lines)
                start_line = i - len(current_chunk_lines)
                chunk_index += 1
            
            current_chunk_lines.append(line)
            current_size += line_size
        
        # Create final chunk
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            chunks.append(LogChunk.create(
                index=chunk_index,
                content=chunk_content,
                start_line=start_line,
                end_line=len(lines) - 1
            ))
        
        return chunks
    
    async def _process_chunk_with_semaphore(
        self,
        chunk: LogChunk,
        state: UnifiedState,
        analyze_func: Any
    ) -> Optional[Dict[str, Any]]:
        """Process a chunk with semaphore control."""
        async with self.processing_semaphore:
            return await self._process_chunk(chunk, state, analyze_func)
    
    async def _process_chunk(
        self,
        chunk: LogChunk,
        state: UnifiedState,
        analyze_func: Any
    ) -> Optional[Dict[str, Any]]:
        """Process a single chunk."""
        try:
            logger.info(f"Processing chunk {chunk.index + 1}/{self.total_chunks}")
            
            # Create chunk-specific state
            chunk_state = UnifiedState(
                log_content=chunk.content,
                log_metadata={
                    **state.log_metadata,
                    "chunk_index": chunk.index,
                    "chunk_hash": chunk.hash,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "is_chunk": True,
                    "total_chunks": self.total_chunks
                },
                enabled_features=state.enabled_features
            )
            
            # Apply rate limiting
            rate_limiter = get_rate_limiter("gemini")
            await rate_limiter.acquire()
            
            # Apply circuit breaker
            circuit_breaker = get_circuit_breaker("chunk_analysis")
            
            # Analyze chunk
            result = await circuit_breaker.async_call(
                analyze_func,
                chunk_state
            )
            
            # Adjust line numbers in issues
            if "issues" in result:
                for issue in result["issues"]:
                    if "line_number" in issue and issue["line_number"] is not None:
                        issue["line_number"] += chunk.start_line
            
            # Store result
            self.processed_chunks[chunk.index] = {
                "chunk_index": chunk.index,
                "chunk_hash": chunk.hash,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                **result
            }
            
            return self.processed_chunks[chunk.index]
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.index}: {str(e)}")
            return None
    
    def _merge_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge results from all chunks into a unified result."""
        merged = {
            "issues": [],
            "suggestions": [],
            "patterns": {},
            "metrics": {},
            "summary": "",
            "chunk_summaries": []
        }
        
        # Track unique issues to avoid duplicates
        seen_issues = set()
        
        for result in sorted(chunk_results, key=lambda x: x.get("chunk_index", 0)):
            # Merge issues
            for issue in result.get("issues", []):
                # Create issue signature for deduplication
                issue_sig = (
                    issue.get("type", ""),
                    issue.get("description", "")[:50],
                    issue.get("severity", "")
                )
                
                if issue_sig not in seen_issues:
                    seen_issues.add(issue_sig)
                    merged["issues"].append(issue)
            
            # Merge suggestions (deduplicate by suggestion text)
            for suggestion in result.get("suggestions", []):
                if not any(s["suggestion"] == suggestion["suggestion"] 
                          for s in merged["suggestions"]):
                    merged["suggestions"].append(suggestion)
            
            # Merge patterns
            for pattern_type, patterns in result.get("patterns", {}).items():
                if pattern_type not in merged["patterns"]:
                    merged["patterns"][pattern_type] = []
                merged["patterns"][pattern_type].extend(patterns)
            
            # Collect chunk summaries
            if result.get("summary"):
                merged["chunk_summaries"].append({
                    "chunk_index": result.get("chunk_index"),
                    "summary": result.get("summary")
                })
        
        # Sort issues by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        merged["issues"].sort(
            key=lambda x: (severity_order.get(x.get("severity", "low"), 4), x.get("line_number", 0))
        )
        
        # Create overall summary
        total_issues = len(merged["issues"])
        critical_issues = sum(1 for i in merged["issues"] if i.get("severity") == "critical")
        high_issues = sum(1 for i in merged["issues"] if i.get("severity") == "high")
        
        merged["summary"] = (
            f"Analyzed {len(chunk_results)} chunks. "
            f"Found {total_issues} issues "
            f"({critical_issues} critical, {high_issues} high severity). "
            f"Processing completed in {self.end_time - self.start_time:.2f} seconds."
        )
        
        # Add streaming metrics
        merged["streaming_metrics"] = {
            "total_chunks": self.total_chunks,
            "processed_chunks": len(chunk_results),
            "chunk_size_mb": self.chunk_size_mb,
            "parallel_chunks": self.max_concurrent_chunks,
            "processing_time": self.end_time - self.start_time if self.end_time else None
        }
        
        return merged


class StreamingAnalyzer:
    """High-level streaming analyzer that coordinates the streaming process."""
    
    def __init__(self):
        self.processor = StreamingProcessor()
    
    async def analyze_with_streaming(
        self,
        log_content: str,
        state: UnifiedState,
        base_analyze_func: Any
    ) -> Dict[str, Any]:
        """Analyze log content using streaming if appropriate.
        
        Args:
            log_content: Log content to analyze
            state: Unified state
            base_analyze_func: Base analysis function
            
        Returns:
            Analysis results
        """
        log_size_mb = len(log_content.encode()) / (1024 * 1024)
        
        # Determine if streaming is needed
        if log_size_mb > 10 or state.has_feature("streaming"):
            logger.info(f"Using streaming analysis for {log_size_mb:.2f}MB log")
            
            # Collect all streaming results
            final_result = None
            async for result in self.processor.process_log_stream(
                log_content, state, base_analyze_func
            ):
                if result["type"] == "stream_complete":
                    final_result = result["final_result"]
                elif result["type"] == "chunk_complete":
                    # Could yield intermediate results here for real-time updates
                    logger.info(f"Chunk {result['chunk_index']} complete: "
                              f"{result['issues_found']} issues found")
            
            return final_result
        else:
            # Use regular analysis for small logs
            logger.info(f"Using regular analysis for {log_size_mb:.2f}MB log")
            return await base_analyze_func(state)


# Global streaming analyzer
_streaming_analyzer = StreamingAnalyzer()


def get_streaming_analyzer() -> StreamingAnalyzer:
    """Get the global streaming analyzer."""
    return _streaming_analyzer