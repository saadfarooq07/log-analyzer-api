#!/usr/bin/env python3
"""Simple test for the LangGraph workflow."""

import asyncio
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_simple():
    """Test just the analysis node directly."""
    
    # Import after loading env
    from app.agent.nodes import analyze_logs
    from app.agent.state import State
    
    test_state = {
        "log_content": """
2024-01-20 10:15:23 ERROR [database] Connection timeout after 30s
2024-01-20 10:15:24 ERROR [database] Failed to connect to PostgreSQL on localhost:5432
2024-01-20 10:15:25 WARN [api] Fallback to cache due to database error
""",
        "environment_details": {"os": "Ubuntu 22.04"},
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
    
    print("🧪 Testing Analysis Node...")
    print("=" * 40)
    
    try:
        # Test just the analysis node
        result = await analyze_logs(test_state)
        
        if result.get("error"):
            print(f"❌ Error: {result['error']}")
        elif result.get("current_analysis"):
            analysis = result["current_analysis"]
            print("✅ Analysis completed!")
            print(f"Issues found: {len(analysis.get('issues', []))}")
            for issue in analysis.get("issues", []):
                print(f"  - [{issue.get('severity', 'UNKNOWN').upper()}] {issue.get('description', '')}")
        else:
            print("❌ No analysis result")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple())