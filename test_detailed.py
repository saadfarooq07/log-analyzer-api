#!/usr/bin/env python3
"""Detailed test to debug the analysis."""

import asyncio
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

async def test_detailed():
    """Test the model response directly."""
    
    from app.config import settings
    from app.agent.prompts import SYSTEM_PROMPT, ANALYSIS_PROMPT
    
    model = ChatGroq(
        model="llama3-8b-8192",
        groq_api_key=settings.groq_api_key,
        temperature=0.7,
        max_tokens=4096
    )
    
    log_content = """
2024-01-20 10:15:23 ERROR [database] Connection timeout after 30s
2024-01-20 10:15:24 ERROR [database] Failed to connect to PostgreSQL on localhost:5432
2024-01-20 10:15:25 WARN [api] Fallback to cache due to database error
2024-01-20 10:15:30 ERROR [database] Connection pool exhausted
"""
    
    analysis_prompt = ANALYSIS_PROMPT.format(
        log_content=log_content,
        environment_details=json.dumps({"os": "Ubuntu 22.04"}, indent=2),
        application_name="web-api",
        analysis_type="general"
    )
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=analysis_prompt)
    ]
    
    print("🧪 Testing Model Response...")
    print("=" * 50)
    
    try:
        response = await model.ainvoke(messages)
        print("Raw Response:")
        print(response.content)
        print("\n" + "="*50)
        
        # Try to parse JSON
        content = response.content
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        
        try:
            parsed = json.loads(content)
            print("✅ Successfully parsed JSON")
            print(f"Issues found: {len(parsed.get('issues', []))}")
            for issue in parsed.get('issues', []):
                print(f"  - {issue}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print("Attempting fallback parsing...")
            
    except Exception as e:
        print(f"❌ Model error: {e}")

if __name__ == "__main__":
    asyncio.run(test_detailed())