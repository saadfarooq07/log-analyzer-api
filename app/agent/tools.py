"""Tools for the log analyzer agent."""

from typing import Dict, Any, List
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
import os
import logging

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