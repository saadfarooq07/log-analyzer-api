"""Tools for the log analyzer agent with full feature support."""

from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
import os
import logging
import json
import re
from datetime import datetime

logger = logging.getLogger(__name__)


@tool
def search_documentation(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Search for relevant documentation and solutions online.
    
    Args:
        query: Search query related to the log issue
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, url, and content
    """
    try:
        # Initialize Tavily search
        search = TavilySearchResults(
            api_key=os.getenv("TAVILY_API_KEY"),
            max_results=max_results
        )
        
        # Enhance query for better technical results
        enhanced_query = f"{query} documentation solution fix"
        
        # Perform search
        results = search.run(enhanced_query)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "relevance": "high" if query.lower() in result.get("content", "").lower() else "medium"
            })
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Documentation search error: {str(e)}")
        return [{
            "title": "Search Error",
            "url": "",
            "content": f"Unable to search documentation: {str(e)}",
            "relevance": "low"
        }]


@tool
def extract_patterns(log_content: str, pattern_type: str = "error") -> List[Dict[str, Any]]:
    """Extract common patterns from log content.
    
    Args:
        log_content: The log content to analyze
        pattern_type: Type of pattern to extract (error, warning, info)
        
    Returns:
        List of identified patterns with count and examples
    """
    patterns = []
    pattern_counts = {}
    
    # Common log patterns to look for
    if pattern_type == "error":
        keywords = ["error", "exception", "failed", "failure", "fatal", "critical"]
    elif pattern_type == "warning":
        keywords = ["warning", "warn", "deprecated", "timeout"]
    else:
        keywords = ["info", "debug", "trace"]
    
    lines = log_content.splitlines()
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for keyword in keywords:
            if keyword in line_lower:
                # Extract a pattern signature (first 50 chars after keyword)
                pattern_start = line_lower.find(keyword)
                pattern_sig = line[pattern_start:pattern_start + 50].strip()
                
                if pattern_sig not in pattern_counts:
                    pattern_counts[pattern_sig] = {
                        "count": 0,
                        "examples": [],
                        "line_numbers": []
                    }
                
                pattern_counts[pattern_sig]["count"] += 1
                if len(pattern_counts[pattern_sig]["examples"]) < 3:
                    pattern_counts[pattern_sig]["examples"].append(line.strip())
                pattern_counts[pattern_sig]["line_numbers"].append(i + 1)
    
    # Convert to list format
    for pattern, data in pattern_counts.items():
        patterns.append({
            "pattern": pattern,
            "type": pattern_type,
            "count": data["count"],
            "examples": data["examples"],
            "first_occurrence": data["line_numbers"][0] if data["line_numbers"] else None,
            "severity": "high" if data["count"] > 10 else "medium" if data["count"] > 5 else "low"
        })
    
    # Sort by count descending
    patterns.sort(key=lambda x: x["count"], reverse=True)
    
    return patterns[:10]  # Return top 10 patterns


@tool
def generate_diagnostic_commands(issue_type: str, platform: str = "linux") -> List[Dict[str, str]]:
    """Generate diagnostic commands based on issue type.
    
    Args:
        issue_type: Type of issue (database, network, disk, memory, etc.)
        platform: Operating system platform
        
    Returns:
        List of diagnostic commands with descriptions
    """
    commands = []
    
    # Common diagnostic commands by issue type
    command_map = {
        "database": {
            "linux": [
                {"command": "netstat -an | grep :5432", "description": "Check PostgreSQL connections"},
                {"command": "ps aux | grep postgres", "description": "Check PostgreSQL processes"},
                {"command": "df -h", "description": "Check disk space"},
                {"command": "tail -n 100 /var/log/postgresql/*.log", "description": "Check database logs"}
            ],
            "windows": [
                {"command": "netstat -an | findstr :5432", "description": "Check PostgreSQL connections"},
                {"command": "tasklist | findstr postgres", "description": "Check PostgreSQL processes"},
                {"command": "wmic logicaldisk get size,freespace,caption", "description": "Check disk space"}
            ]
        },
        "network": {
            "linux": [
                {"command": "netstat -tuln", "description": "Check listening ports"},
                {"command": "ss -s", "description": "Socket statistics"},
                {"command": "ping -c 4 8.8.8.8", "description": "Test internet connectivity"},
                {"command": "traceroute google.com", "description": "Trace network path"}
            ],
            "windows": [
                {"command": "netstat -an", "description": "Check network connections"},
                {"command": "ping -n 4 8.8.8.8", "description": "Test internet connectivity"},
                {"command": "tracert google.com", "description": "Trace network path"}
            ]
        },
        "memory": {
            "linux": [
                {"command": "free -h", "description": "Check memory usage"},
                {"command": "top -b -n 1 | head -20", "description": "Check top processes"},
                {"command": "vmstat 1 5", "description": "Virtual memory statistics"},
                {"command": "ps aux --sort=-%mem | head", "description": "Top memory consumers"}
            ],
            "windows": [
                {"command": "wmic OS get TotalVisibleMemorySize,FreePhysicalMemory", "description": "Check memory"},
                {"command": "tasklist /FO TABLE", "description": "List all processes"},
                {"command": "wmic process get Name,WorkingSetSize", "description": "Process memory usage"}
            ]
        },
        "disk": {
            "linux": [
                {"command": "df -h", "description": "Check disk usage"},
                {"command": "du -sh /* | sort -h", "description": "Find large directories"},
                {"command": "iostat -x 1 5", "description": "Disk I/O statistics"},
                {"command": "lsof | grep deleted", "description": "Find deleted files still open"}
            ],
            "windows": [
                {"command": "wmic logicaldisk get size,freespace,caption", "description": "Check disk space"},
                {"command": "dir C:\\ /s | sort", "description": "List directory sizes"}
            ]
        }
    }
    
    # Get commands for the issue type
    if issue_type.lower() in command_map:
        platform_commands = command_map[issue_type.lower()].get(platform.lower(), [])
        commands.extend(platform_commands)
    
    # Add general diagnostic commands
    general_commands = [
        {"command": "date", "description": "Check system time"},
        {"command": "uptime", "description": "Check system uptime"} if platform == "linux" else {"command": "systeminfo | findstr /C:\"System Boot Time\"", "description": "Check system uptime"}
    ]
    commands.extend(general_commands)
    
    return commands


@tool
def request_additional_info(question: str, context: str = "") -> Dict[str, Any]:
    """Request additional information from the user for interactive analysis.
    
    Args:
        question: The question to ask the user
        context: Additional context about why this information is needed
        
    Returns:
        Dictionary indicating that user interaction is required
    """
    return {
        "type": "user_interaction_required",
        "question": question,
        "context": context,
        "timestamp": datetime.now().isoformat(),
        "requires_response": True
    }


@tool
def submit_analysis(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """Submit the final analysis result.
    
    Args:
        analysis_result: The complete analysis result to submit
        
    Returns:
        Confirmation of submission
    """
    return {
        "type": "analysis_submitted",
        "result": analysis_result,
        "timestamp": datetime.now().isoformat(),
        "status": "completed"
    }


@tool
def analyze_log_patterns(log_content: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """Perform advanced pattern analysis on log content.
    
    Args:
        log_content: The log content to analyze
        analysis_type: Type of analysis (comprehensive, security, performance, error)
        
    Returns:
        Advanced pattern analysis results
    """
    try:
        lines = log_content.splitlines()
        total_lines = len(lines)
        
        # Initialize analysis results
        analysis = {
            "total_lines": total_lines,
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "patterns": {},
            "anomalies": [],
            "trends": {},
            "severity_distribution": {"critical": 0, "high": 0, "medium": 0, "low": 0}
        }
        
        # Pattern detection based on analysis type
        if analysis_type in ["comprehensive", "error"]:
            analysis["patterns"]["errors"] = _extract_error_patterns(lines)
        
        if analysis_type in ["comprehensive", "security"]:
            analysis["patterns"]["security"] = _extract_security_patterns(lines)
        
        if analysis_type in ["comprehensive", "performance"]:
            analysis["patterns"]["performance"] = _extract_performance_patterns(lines)
        
        # Detect anomalies
        analysis["anomalies"] = _detect_anomalies(lines)
        
        # Analyze trends
        analysis["trends"] = _analyze_trends(lines)
        
        # Calculate severity distribution
        for pattern_type in analysis["patterns"]:
            for pattern in analysis["patterns"][pattern_type]:
                severity = pattern.get("severity", "low")
                analysis["severity_distribution"][severity] += pattern.get("count", 0)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Pattern analysis error: {str(e)}")
        return {
            "error": f"Pattern analysis failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


def _extract_error_patterns(lines: List[str]) -> List[Dict[str, Any]]:
    """Extract error patterns from log lines."""
    error_patterns = []
    error_keywords = ["error", "exception", "failed", "failure", "fatal", "critical", "panic"]
    
    pattern_counts = {}
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for keyword in error_keywords:
            if keyword in line_lower:
                # Extract error signature
                error_match = re.search(rf'{keyword}[:\s]*([^,\n\r]+)', line, re.IGNORECASE)
                if error_match:
                    error_sig = error_match.group(1).strip()[:100]  # Limit length
                    
                    if error_sig not in pattern_counts:
                        pattern_counts[error_sig] = {
                            "count": 0,
                            "first_line": i + 1,
                            "last_line": i + 1,
                            "examples": [],
                            "keyword": keyword
                        }
                    
                    pattern_counts[error_sig]["count"] += 1
                    pattern_counts[error_sig]["last_line"] = i + 1
                    
                    if len(pattern_counts[error_sig]["examples"]) < 3:
                        pattern_counts[error_sig]["examples"].append(line.strip())
    
    # Convert to list format
    for pattern, data in pattern_counts.items():
        severity = "critical" if data["count"] > 50 else "high" if data["count"] > 10 else "medium"
        error_patterns.append({
            "pattern": pattern,
            "type": "error",
            "keyword": data["keyword"],
            "count": data["count"],
            "severity": severity,
            "first_occurrence": data["first_line"],
            "last_occurrence": data["last_line"],
            "examples": data["examples"]
        })
    
    return sorted(error_patterns, key=lambda x: x["count"], reverse=True)


def _extract_security_patterns(lines: List[str]) -> List[Dict[str, Any]]:
    """Extract security-related patterns from log lines."""
    security_patterns = []
    security_keywords = [
        "authentication failed", "login failed", "unauthorized", "access denied",
        "permission denied", "forbidden", "invalid credentials", "brute force",
        "suspicious", "malicious", "intrusion", "attack"
    ]
    
    pattern_counts = {}
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for keyword in security_keywords:
            if keyword in line_lower:
                # Extract IP addresses if present
                ip_match = re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', line)
                ip_addr = ip_match.group(0) if ip_match else "unknown"
                
                pattern_key = f"{keyword}_{ip_addr}"
                
                if pattern_key not in pattern_counts:
                    pattern_counts[pattern_key] = {
                        "count": 0,
                        "keyword": keyword,
                        "ip_address": ip_addr,
                        "first_line": i + 1,
                        "examples": []
                    }
                
                pattern_counts[pattern_key]["count"] += 1
                if len(pattern_counts[pattern_key]["examples"]) < 2:
                    pattern_counts[pattern_key]["examples"].append(line.strip())
    
    # Convert to list format
    for pattern_key, data in pattern_counts.items():
        severity = "critical" if data["count"] > 20 else "high" if data["count"] > 5 else "medium"
        security_patterns.append({
            "pattern": data["keyword"],
            "type": "security",
            "ip_address": data["ip_address"],
            "count": data["count"],
            "severity": severity,
            "first_occurrence": data["first_line"],
            "examples": data["examples"]
        })
    
    return sorted(security_patterns, key=lambda x: x["count"], reverse=True)


def _extract_performance_patterns(lines: List[str]) -> List[Dict[str, Any]]:
    """Extract performance-related patterns from log lines."""
    performance_patterns = []
    perf_keywords = ["timeout", "slow", "latency", "response time", "memory", "cpu", "disk"]
    
    # Look for timing patterns
    timing_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(ms|seconds?|s)\b', re.IGNORECASE)
    
    pattern_counts = {}
    timing_values = []
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Check for performance keywords
        for keyword in perf_keywords:
            if keyword in line_lower:
                if keyword not in pattern_counts:
                    pattern_counts[keyword] = {
                        "count": 0,
                        "first_line": i + 1,
                        "examples": []
                    }
                
                pattern_counts[keyword]["count"] += 1
                if len(pattern_counts[keyword]["examples"]) < 2:
                    pattern_counts[keyword]["examples"].append(line.strip())
        
        # Extract timing values
        timing_matches = timing_pattern.findall(line)
        for value, unit in timing_matches:
            # Convert to milliseconds
            ms_value = float(value)
            if unit.lower() in ['s', 'seconds', 'second']:
                ms_value *= 1000
            
            timing_values.append({
                "value": ms_value,
                "line": i + 1,
                "original": f"{value} {unit}"
            })
    
    # Convert keyword patterns to list format
    for keyword, data in pattern_counts.items():
        severity = "high" if data["count"] > 20 else "medium" if data["count"] > 5 else "low"
        performance_patterns.append({
            "pattern": keyword,
            "type": "performance",
            "count": data["count"],
            "severity": severity,
            "first_occurrence": data["first_line"],
            "examples": data["examples"]
        })
    
    # Add timing analysis if we found timing values
    if timing_values:
        avg_time = sum(t["value"] for t in timing_values) / len(timing_values)
        max_time = max(timing_values, key=lambda x: x["value"])
        
        performance_patterns.append({
            "pattern": "response_times",
            "type": "performance_timing",
            "count": len(timing_values),
            "severity": "high" if avg_time > 5000 else "medium" if avg_time > 1000 else "low",
            "average_ms": avg_time,
            "max_time": max_time,
            "examples": [f"Average: {avg_time:.1f}ms", f"Max: {max_time['original']} at line {max_time['line']}"]
        })
    
    return sorted(performance_patterns, key=lambda x: x["count"], reverse=True)


def _detect_anomalies(lines: List[str]) -> List[Dict[str, Any]]:
    """Detect anomalies in log patterns."""
    anomalies = []
    
    # Check for sudden spikes in log volume
    if len(lines) > 1000:
        # Sample every 100 lines to check for volume spikes
        volume_samples = []
        for i in range(0, len(lines), 100):
            chunk = lines[i:i+100]
            volume_samples.append(len(chunk))
        
        if volume_samples:
            avg_volume = sum(volume_samples) / len(volume_samples)
            for i, volume in enumerate(volume_samples):
                if volume > avg_volume * 3:  # 3x spike
                    anomalies.append({
                        "type": "volume_spike",
                        "description": f"Log volume spike detected around line {i*100}",
                        "severity": "medium",
                        "line_range": [i*100, (i+1)*100],
                        "volume": volume,
                        "average": avg_volume
                    })
    
    # Check for repeated identical messages (potential loops)
    line_counts = {}
    for i, line in enumerate(lines):
        # Normalize line (remove timestamps, etc.)
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}[\s\d:,-]+', '', line)
        normalized = re.sub(r'\b\d+\b', 'N', normalized)  # Replace numbers with N
        
        if normalized not in line_counts:
            line_counts[normalized] = {"count": 0, "lines": []}
        
        line_counts[normalized]["count"] += 1
        line_counts[normalized]["lines"].append(i + 1)
    
    # Find repeated messages
    for normalized_line, data in line_counts.items():
        if data["count"] > 50:  # Repeated more than 50 times
            anomalies.append({
                "type": "repeated_message",
                "description": f"Message repeated {data['count']} times",
                "severity": "high" if data["count"] > 200 else "medium",
                "pattern": normalized_line[:100],
                "count": data["count"],
                "first_line": data["lines"][0],
                "last_line": data["lines"][-1]
            })
    
    return anomalies


def _analyze_trends(lines: List[str]) -> Dict[str, Any]:
    """Analyze trends in the log data."""
    trends = {
        "timestamp": datetime.now().isoformat(),
        "log_volume_trend": "stable",
        "error_trend": "stable",
        "time_analysis": {}
    }
    
    # Simple trend analysis based on line distribution
    if len(lines) > 100:
        # Split into 4 quarters
        quarter_size = len(lines) // 4
        quarters = [
            lines[0:quarter_size],
            lines[quarter_size:quarter_size*2],
            lines[quarter_size*2:quarter_size*3],
            lines[quarter_size*3:]
        ]
        
        # Count errors in each quarter
        error_counts = []
        for quarter in quarters:
            error_count = sum(1 for line in quarter if any(
                keyword in line.lower() 
                for keyword in ["error", "exception", "failed", "fatal"]
            ))
            error_counts.append(error_count)
        
        # Determine error trend
        if len(error_counts) >= 2:
            if error_counts[-1] > error_counts[0] * 1.5:
                trends["error_trend"] = "increasing"
            elif error_counts[-1] < error_counts[0] * 0.5:
                trends["error_trend"] = "decreasing"
        
        trends["error_distribution"] = error_counts
    
    return trends