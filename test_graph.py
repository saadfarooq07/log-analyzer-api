#!/usr/bin/env python3
"""Test the LangGraph workflow locally."""

import asyncio
import json
from dotenv import load_dotenv
from app.agent.graph import graph

# Load environment variables
load_dotenv()

async def test_graph():
    """Test the graph with sample log data."""
    
    test_input = {
        "log_content": """
2024-01-20 10:15:23 ERROR [database] Connection timeout after 30s
2024-01-20 10:15:24 ERROR [database] Failed to connect to PostgreSQL on localhost:5432
2024-01-20 10:15:25 WARN [api] Fallback to cache due to database error
2024-01-20 10:15:30 ERROR [database] Connection pool exhausted
""",
        "environment_details": {
            "os": "Ubuntu 22.04",
            "postgresql_version": "14.5"
        },
        "application_name": "web-api",
        "analysis_type": "general",
        "include_suggestions": True,
        "include_documentation": True,
        "messages": [],
        "current_analysis": None,
        "validation_result": None,
        "tool_calls": [],
        "analysis_result": None,
        "error": None
    }
    
    print("🧪 Testing Log Analyzer Graph...")
    print("=" * 50)
    
    try:
        # Run the graph
        result = await graph.ainvoke(test_input)
        
        # Print results
        if result.get("analysis_result"):
            analysis = result["analysis_result"]
            print("\n✅ Analysis completed!")
            print(f"\nIssues found: {len(analysis.get('issues', []))}")
            
            for issue in analysis.get("issues", []):
                print(f"  - [{issue.get('severity', 'UNKNOWN').upper()}] {issue.get('description', '')}")
            
            if analysis.get("suggestions"):
                print(f"\nSuggestions: {len(analysis.get('suggestions', []))}")
                for sug in analysis.get("suggestions", [])[:3]:
                    print(f"  - {sug.get('suggestion', '')}")
            
            print(f"\nSummary: {analysis.get('summary', 'No summary available')}")
        else:
            print("❌ No analysis result returned")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error running graph: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_graph())