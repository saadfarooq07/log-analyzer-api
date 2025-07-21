"""Node implementations for the log analyzer workflow."""

import json
import logging
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.agent.state import State
from app.agent.prompts import SYSTEM_PROMPT, ANALYSIS_PROMPT, VALIDATION_PROMPT, STREAMING_CHUNK_PROMPT
from app.agent.tools import search_documentation, extract_patterns, generate_diagnostic_commands
from app.config import settings

logger = logging.getLogger(__name__)


async def analyze_logs(state: State) -> Dict[str, Any]:
    """Main node for analyzing log content."""
    try:
        # Initialize the primary model (Gemini)
        model = ChatGoogleGenerativeAI(
            model=settings.primary_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.temperature,
            max_output_tokens=settings.max_tokens
        )
        
        # Check if we need to chunk for large logs
        log_size_mb = len(state["log_content"].encode()) / (1024 * 1024)
        
        if log_size_mb > 5:  # Use chunking for logs > 5MB
            logger.info(f"Using chunked analysis for {log_size_mb:.2f}MB log")
            analysis_result = await analyze_large_log_chunked(state, model)
        else:
            # Prepare the analysis prompt
            analysis_prompt = ANALYSIS_PROMPT.format(
                log_content=state["log_content"][:50000],  # Limit to first 50k chars for safety
                environment_details=json.dumps(state["environment_details"], indent=2),
                application_name=state.get("application_name", "Unknown"),
                analysis_type=state.get("analysis_type", "general")
            )
            
            # Create messages
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=analysis_prompt)
            ]
            
            # Get analysis from model
            response = await model.ainvoke(messages)
            
            # Parse the response
            try:
                # Try to extract JSON from the response
                content = response.content
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    content = content[json_start:json_end].strip()
                
                analysis_result = json.loads(content)
            except json.JSONDecodeError:
                # Fallback: create structured result from text
                analysis_result = parse_unstructured_analysis(response.content)
        
        # Extract patterns for additional insights
        error_patterns = extract_patterns(state["log_content"], "error")
        warning_patterns = extract_patterns(state["log_content"], "warning")
        
        # Enhance issues with pattern information
        if "issues" not in analysis_result:
            analysis_result["issues"] = []
        
        # Add pattern-based issues
        for pattern in error_patterns[:5]:  # Top 5 error patterns
            if pattern["count"] > 3:  # Repeated errors are significant
                analysis_result["issues"].append({
                    "type": "error_pattern",
                    "description": f"Repeated error pattern: {pattern['pattern']} (occurred {pattern['count']} times)",
                    "severity": pattern["severity"],
                    "line_number": pattern["first_occurrence"],
                    "pattern": pattern["pattern"]
                })
        
        # Search for documentation if requested
        if state.get("include_documentation", True) and analysis_result.get("issues"):
            # Search for the most critical issue
            critical_issues = [i for i in analysis_result["issues"] if i.get("severity") in ["critical", "high"]]
            if critical_issues:
                search_query = critical_issues[0]["description"]
                doc_results = search_documentation(search_query, max_results=3)
                
                if "documentation_references" not in analysis_result:
                    analysis_result["documentation_references"] = []
                
                for doc in doc_results:
                    analysis_result["documentation_references"].append({
                        "title": doc["title"],
                        "url": doc["url"],
                        "relevance": doc["relevance"],
                        "excerpt": doc["content"][:200] + "..."
                    })
        
        # Generate diagnostic commands
        if analysis_result.get("issues"):
            issue_types = set()
            for issue in analysis_result["issues"]:
                issue_type = issue.get("type", "").lower()
                if "database" in issue_type or "connection" in issue_type:
                    issue_types.add("database")
                elif "network" in issue_type or "timeout" in issue_type:
                    issue_types.add("network")
                elif "memory" in issue_type or "oom" in issue_type:
                    issue_types.add("memory")
                elif "disk" in issue_type or "space" in issue_type:
                    issue_types.add("disk")
            
            if "diagnostic_commands" not in analysis_result:
                analysis_result["diagnostic_commands"] = []
            
            for issue_type in issue_types:
                commands = generate_diagnostic_commands(issue_type, "linux")
                analysis_result["diagnostic_commands"].extend(commands[:3])  # Top 3 commands per type
        
        # Update state
        state["current_analysis"] = analysis_result
        state["messages"].append(AIMessage(content=json.dumps(analysis_result, indent=2)))
        
        logger.info(f"Analysis completed: {len(analysis_result.get('issues', []))} issues found")
        
        return state
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        state["error"] = str(e)
        return state


async def analyze_large_log_chunked(state: State, model) -> Dict[str, Any]:
    """Analyze large logs in chunks."""
    log_content = state["log_content"]
    lines = log_content.splitlines()
    chunk_size = 1000  # Lines per chunk
    total_chunks = (len(lines) + chunk_size - 1) // chunk_size
    
    all_issues = []
    all_suggestions = []
    
    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i:i + chunk_size]
        chunk_content = "\n".join(chunk_lines)
        
        chunk_prompt = STREAMING_CHUNK_PROMPT.format(
            chunk_number=(i // chunk_size) + 1,
            total_chunks=total_chunks,
            log_chunk=chunk_content
        )
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=chunk_prompt)
        ]
        
        response = await model.ainvoke(messages)
        
        try:
            # Parse chunk result
            content = response.content
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            
            chunk_result = json.loads(content)
            
            # Adjust line numbers for chunk offset
            for issue in chunk_result.get("issues", []):
                if issue.get("line_number"):
                    issue["line_number"] += i
            
            all_issues.extend(chunk_result.get("issues", []))
            all_suggestions.extend(chunk_result.get("suggestions", []))
            
        except Exception as e:
            logger.error(f"Error parsing chunk {i // chunk_size + 1}: {str(e)}")
    
    # Deduplicate and merge results
    unique_issues = []
    seen_descriptions = set()
    for issue in all_issues:
        desc = issue.get("description", "")
        if desc not in seen_descriptions:
            seen_descriptions.add(desc)
            unique_issues.append(issue)
    
    return {
        "issues": unique_issues,
        "suggestions": list({s["suggestion"]: s for s in all_suggestions}.values()),  # Dedupe suggestions
        "summary": f"Analyzed {total_chunks} chunks, found {len(unique_issues)} unique issues"
    }


async def validate_analysis(state: State) -> Dict[str, Any]:
    """Validate the analysis results."""
    try:
        # Use orchestration model for validation
        model = ChatGroq(
            model="mixtral-8x7b-32768",  # Using Mixtral as Kimi K2 might not be available
            groq_api_key=settings.groq_api_key,
            temperature=0.3  # Lower temperature for validation
        )
        
        current_analysis = state.get("current_analysis", {})
        
        # Prepare validation prompt
        validation_prompt = VALIDATION_PROMPT.format(
            current_analysis=json.dumps(current_analysis, indent=2)
        )
        
        messages = [
            SystemMessage(content="You are a quality assurance specialist for log analysis."),
            HumanMessage(content=validation_prompt)
        ]
        
        # Get validation result
        response = await model.ainvoke(messages)
        
        # Parse validation feedback
        validation_content = response.content.lower()
        
        # Simple validation logic
        is_complete = any(word in validation_content for word in ["complete", "thorough", "comprehensive", "good", "accurate"])
        needs_improvement = any(word in validation_content for word in ["missing", "incomplete", "improve", "add", "insufficient"])
        
        validation_result = {
            "is_valid": is_complete and not needs_improvement,
            "feedback": response.content,
            "confidence": 0.9 if is_complete else 0.5
        }
        
        state["validation_result"] = validation_result
        
        # If valid, set the final analysis result
        if validation_result["is_valid"]:
            state["analysis_result"] = current_analysis
            logger.info("Analysis validated successfully")
        else:
            logger.warning(f"Analysis needs improvement: {validation_result['feedback'][:100]}...")
        
        return state
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        # If validation fails, accept the analysis anyway
        state["analysis_result"] = state.get("current_analysis", {})
        return state


def parse_unstructured_analysis(text: str) -> Dict[str, Any]:
    """Parse unstructured text analysis into structured format."""
    # Basic parsing logic for fallback
    lines = text.split("\n")
    issues = []
    suggestions = []
    
    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect sections
        if "issue" in line.lower() or "error" in line.lower() or "problem" in line.lower():
            current_section = "issues"
        elif "suggestion" in line.lower() or "fix" in line.lower() or "solution" in line.lower():
            current_section = "suggestions"
        elif current_section == "issues" and line.startswith("-"):
            issues.append({
                "type": "general",
                "description": line[1:].strip(),
                "severity": "medium"
            })
        elif current_section == "suggestions" and line.startswith("-"):
            suggestions.append({
                "issue_type": "general",
                "suggestion": line[1:].strip(),
                "priority": "medium"
            })
    
    return {
        "issues": issues,
        "suggestions": suggestions,
        "summary": "Analysis completed (parsed from unstructured response)"
    }