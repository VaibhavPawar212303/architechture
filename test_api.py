#!/usr/bin/env python3
"""
Test script for Phi-3 FastAPI server
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to the API. Is the container running?")
        return False

def wait_for_model():
    """Wait for model to load"""
    print("⏳ Waiting for model to load...")
    max_retries = 20
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                data = response.json()
                if data.get("model_loaded"):
                    print("✅ Model loaded successfully!")
                    return True
            
            print(f"⏳ Model still loading... ({retry_count + 1}/{max_retries})")
            time.sleep(10)
            retry_count += 1
            
        except requests.exceptions.ConnectionError:
            print(f"⏳ Waiting for container to start... ({retry_count + 1}/{max_retries})")
            time.sleep(5)
            retry_count += 1
    
    print("❌ Timeout waiting for model to load")
    return False

def test_chat():
    """Test chat endpoint"""
    print("\n🔍 Testing chat endpoint...")
    
    message = "Hello! How are you?"
    print(f"\n📝 Testing message: '{message}'")
    
    payload = {
        "message": message,
        "max_length": 256,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat", 
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response: {data['response']}")
        else:
            print(f"❌ Chat failed: {response.status_code} - {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏳ Request timed out - model might still be loading")
    except Exception as e:
        print(f"❌ Error in chat: {e}")

def main():
    print("🧪 Testing Phi-3 FastAPI Server")
    print("=" * 50)
    
    if not wait_for_model():
        print("❌ Cannot proceed - model failed to load")
        return
    
    if not test_health():
        return
    
    test_chat()
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    main()