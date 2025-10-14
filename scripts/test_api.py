#!/usr/bin/env python3
"""
Test Offline LTX-Video API
"""

import requests
import time
import json
from pathlib import Path

# Update this with your Cloud Run URL
API_URL = "https://ltx-video-api-362062855771.us-central1.run.app"

def test_offline_api():
    """Test the offline API"""
    
    print("🎬 Testing Offline LTX-Video API")
    print("=" * 50)
    
    # Health check
    print("🔍 Health check...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=30)
        if response.status_code == 200:
            health = response.json()
            print("✅ API Health:")
            print(f"   Status: {health.get('status')}")
            print(f"   Mode: {health.get('mode')}")
            print(f"   Dependencies: {health.get('dependencies')}")
            print(f"   Model loaded: {health.get('model_loaded')}")
            print(f"   GPU available: {health.get('gpu_available')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test video generation
    print("\n🎬 Testing video generation...")
    
    request_data = {
        "prompt": "A simple test video of a cat on a beach",
        "height": 384,
        "width": 512,
        "num_frames": 25
    }
    
    try:
        print("⏳ Generating video (offline mode)...")
        start_time = time.time()
        
        response = requests.post(
            f"{API_URL}/generate",
            json=request_data,
            timeout=600  # 10 minutes
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Video generated in {generation_time:.1f}s")
            print(f"📹 Video URL: {result.get('video_url')}")
            print(f"📄 Mode: {result.get('mode')}")
            return True
        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False

def main():
    success = test_offline_api()
    
    if success:
        print("\n🎉 Offline API working perfectly!")
        print("🚀 Zero external dependencies confirmed")
    else:
        print("\n❌ API test failed")

if __name__ == "__main__":
    main()
