#!/usr/bin/env python3
"""Example client for LangGraph Cloud SaaS deployment."""

import asyncio
import os
from typing import Dict, Any
from langgraph_sdk import get_client

# Configuration
LANGGRAPH_API_URL = os.getenv("LANGGRAPH_API_URL", "https://YOUR_DEPLOYMENT_ID.api.langchain.com")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "your-langsmith-api-key")


async def analyze_logs(log_content: str, **kwargs) -> Dict[str, Any]:
    """Analyze logs using LangGraph Cloud deployment."""
    
    # Initialize client
    client = get_client(url=LANGGRAPH_API_URL, api_key=LANGSMITH_API_KEY)
    
    # Create a thread
    thread = await client.threads.create()
    print(f"Created thread: {thread['thread_id']}")
    
    # Prepare input
    input_data = {
        "log_content": log_content,
        "environment_details": kwargs.get("environment_details", {}),
        "application_name": kwargs.get("application_name", "unknown"),
        "analysis_type": kwargs.get("analysis_type", "general"),
        "include_suggestions": kwargs.get("include_suggestions", True),
        "include_documentation": kwargs.get("include_documentation", True),
        # Required state fields
        "messages": [],
        "current_analysis": None,
        "validation_result": None,
        "tool_calls": [],
        "analysis_result": None,
        "error": None
    }
    
    # Run analysis
    print("Starting analysis...")
    run = await client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id="log_analyzer",
        input=input_data
    )
    
    # Wait for completion
    print("Waiting for analysis to complete...")
    await client.runs.wait(thread["thread_id"], run["run_id"])
    
    # Get final state
    final_state = await client.threads.get_state(thread["thread_id"])
    
    return final_state["values"]


async def stream_analysis(log_content: str, **kwargs):
    """Stream analysis events in real-time."""
    
    client = get_client(url=LANGGRAPH_API_URL, api_key=LANGSMITH_API_KEY)
    
    thread = await client.threads.create()
    
    input_data = {
        "log_content": log_content,
        "environment_details": kwargs.get("environment_details", {}),
        "application_name": kwargs.get("application_name", "unknown"),
        "analysis_type": kwargs.get("analysis_type", "general"),
        "include_suggestions": True,
        "include_documentation": True,
        "messages": [],
        "current_analysis": None,
        "validation_result": None,
        "tool_calls": [],
        "analysis_result": None,
        "error": None
    }
    
    print("Streaming analysis events...")
    async for event in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id="log_analyzer",
        input=input_data,
        stream_mode="events"
    ):
        if event["event"] == "on_chain_start":
            print(f"🔄 Started: {event['name']}")
        elif event["event"] == "on_chain_end":
            print(f"✅ Completed: {event['name']}")
            if event["name"] == "validate" and "analysis_result" in event.get("data", {}).get("output", {}):
                return event["data"]["output"]["analysis_result"]
        elif event["event"] == "on_tool_start":
            print(f"🔧 Tool: {event['name']}")


async def main():
    """Example usage of the Log Analyzer Cloud client."""
    
    # Sample log data
    test_log = """
2024-01-20 10:15:23 ERROR [database] Connection timeout after 30s
2024-01-20 10:15:24 ERROR [database] Failed to connect to PostgreSQL on localhost:5432
2024-01-20 10:15:25 WARN [api] Fallback to cache due to database error
2024-01-20 10:15:26 INFO [api] Serving cached response
2024-01-20 10:15:30 ERROR [database] Connection pool exhausted
2024-01-20 10:15:31 FATAL [app] Application shutting down due to database issues
"""
    
    print("🚀 Log Analyzer Cloud Client Example")
    print("=" * 50)
    
    # Example 1: Simple analysis
    print("\n📊 Example 1: Simple Analysis")
    result = await analyze_logs(
        test_log,
        environment_details={
            "os": "Ubuntu 22.04",
            "postgresql_version": "14.5"
        },
        application_name="web-api"
    )
    
    if result.get("analysis_result"):
        analysis = result["analysis_result"]
        print(f"\nFound {len(analysis.get('issues', []))} issues:")
        for issue in analysis.get("issues", []):
            print(f"  - [{issue.get('severity', 'UNKNOWN')}] {issue.get('description', '')}")
        
        if analysis.get("suggestions"):
            print(f"\nSuggestions:")
            for sug in analysis.get("suggestions", [])[:3]:
                print(f"  - {sug.get('suggestion', '')}")
    
    # Example 2: Streaming analysis
    print("\n\n📡 Example 2: Streaming Analysis")
    analysis = await stream_analysis(
        test_log,
        environment_details={"os": "Linux"},
        application_name="streaming-test"
    )
    
    if analysis:
        print(f"\nStreaming completed! Found {len(analysis.get('issues', []))} issues")


if __name__ == "__main__":
    # Check configuration
    if LANGGRAPH_API_URL == "https://YOUR_DEPLOYMENT_ID.api.langchain.com":
        print("❌ Please set LANGGRAPH_API_URL environment variable")
        print("   export LANGGRAPH_API_URL=https://your-deployment.api.langchain.com")
        exit(1)
    
    if LANGSMITH_API_KEY == "your-langsmith-api-key":
        print("❌ Please set LANGSMITH_API_KEY environment variable")
        print("   export LANGSMITH_API_KEY=your-actual-api-key")
        exit(1)
    
    asyncio.run(main())