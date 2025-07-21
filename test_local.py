#!/usr/bin/env python3
"""Quick test script for local development."""

import requests
import json
import sys

def test_health():
    """Test health endpoint."""
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"✅ Health check: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_analysis():
    """Test analysis endpoint."""
    test_log = """
2024-01-20 10:15:23 ERROR [database] Connection timeout after 30s
2024-01-20 10:15:24 ERROR [database] Failed to connect to PostgreSQL on localhost:5432
2024-01-20 10:15:25 WARN [api] Fallback to cache due to database error
2024-01-20 10:15:26 INFO [api] Serving cached response
2024-01-20 10:15:30 ERROR [database] Connection pool exhausted
2024-01-20 10:15:31 FATAL [app] Application shutting down due to database issues
"""
    
    data = {
        "log_content": test_log,
        "environment_details": {
            "os": "Ubuntu 22.04",
            "postgresql_version": "14.5",
            "app_version": "2.1.0"
        },
        "application_name": "web-api",
        "analysis_type": "general"
    }
    
    try:
        print("\n📤 Sending analysis request...")
        response = requests.post(
            "http://localhost:8000/api/v1/analyze",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Analysis completed!")
            print(f"   Analysis ID: {result['analysis_id']}")
            print(f"   Issues found: {len(result['issues'])}")
            print(f"   Processing time: {result['metrics']['processing_time']:.2f}s")
            
            print("\n📋 Issues:")
            for issue in result['issues']:
                print(f"   - [{issue['severity'].upper()}] {issue['description']}")
            
            if result.get('suggestions'):
                print("\n💡 Suggestions:")
                for sug in result['suggestions'][:3]:
                    print(f"   - {sug['suggestion']}")
            
            return True
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"   {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Analysis request failed: {e}")
        return False

def main():
    """Run tests."""
    print("🧪 Testing Log Analyzer API...")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("\n⚠️  Server not running? Start with: uvicorn app.main:app --reload")
        sys.exit(1)
    
    # Test analysis
    if test_analysis():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()