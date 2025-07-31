#!/usr/bin/env python3
"""
Simple test script to verify the deployment
"""
import requests
import json

def test_health(base_url):
    """Test health endpoint"""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data.get('status')}")
            print(f"PDF Library: {data.get('pdf_library')}")
            print(f"Vector Search: {data.get('vector_search')}")
            print(f"Gemini Configured: {data.get('gemini_configured')}")
            return True
        else:
            print(f"Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

def test_root(base_url):
    """Test root endpoint"""
    try:
        response = requests.get(base_url, timeout=10)
        print(f"Root endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Message: {data.get('message')}")
            return True
        else:
            print(f"Root endpoint failed: {response.text}")
            return False
    except Exception as e:
        print(f"Root endpoint error: {e}")
        return False

def main():
    """Main test function"""
    # Replace with your actual Render URL
    base_url = "https://your-app-name.onrender.com"
    
    print("Testing LLM Document Processing API...")
    print(f"Base URL: {base_url}")
    print("-" * 50)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    health_ok = test_health(base_url)
    print()
    
    # Test root endpoint
    print("2. Testing root endpoint...")
    root_ok = test_root(base_url)
    print()
    
    # Summary
    print("=" * 50)
    print("TEST SUMMARY:")
    print(f"Health endpoint: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Root endpoint: {'✅ PASS' if root_ok else '❌ FAIL'}")
    
    if health_ok and root_ok:
        print("\n🎉 All tests passed! Deployment is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the deployment logs.")

if __name__ == "__main__":
    main() 