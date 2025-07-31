#!/usr/bin/env python3
"""
Test script to verify deployment
Run this after deployment to ensure everything works
"""

import requests
import json
import os
from typing import Optional

def test_health_check(base_url: str) -> bool:
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_root_endpoint(base_url: str) -> bool:
    """Test the root endpoint"""
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint passed: {data}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_query_endpoint(base_url: str) -> bool:
    """Test the query endpoint with a simple request"""
    try:
        # Test with a simple query (no documents to avoid external dependencies)
        payload = {
            "query": "test query",
            "documents": [],
            "top_k": 3
        }
        
        response = requests.post(
            f"{base_url}/api/v1/query",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 400:  # Expected for empty documents
            print("✅ Query endpoint validation working (correctly rejected empty documents)")
            return True
        elif response.status_code == 200:
            data = response.json()
            print(f"✅ Query endpoint passed: {data}")
            return True
        else:
            print(f"❌ Query endpoint failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Query endpoint error: {e}")
        return False

def main():
    """Main test function"""
    # Get base URL from environment or use default
    base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    
    print(f"🧪 Testing deployment at: {base_url}")
    print("=" * 50)
    
    tests = [
        ("Health Check", lambda: test_health_check(base_url)),
        ("Root Endpoint", lambda: test_root_endpoint(base_url)),
        ("Query Endpoint", lambda: test_query_endpoint(base_url)),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Deployment is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the deployment logs.")
    
    return passed == total

if __name__ == "__main__":
    main() 