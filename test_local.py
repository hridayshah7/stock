#!/usr/bin/env python3
"""
Local test script for the Stock Sniper API
Run this to test the API locally before deploying
"""

import subprocess
import sys
import time
import requests
from datetime import datetime

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    print("✅ Packages installed successfully!")

def test_endpoints():
    """Test API endpoints"""
    base_url = "http://localhost:8000"
    
    print(f"\n🧪 Testing API endpoints at {base_url}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"✅ Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test scan endpoint
    try:
        print("\n🔍 Testing stock scan (this may take a few minutes)...")
        response = requests.get(f"{base_url}/scan?min_score=8&limit=10", timeout=60)
        data = response.json()
        print(f"✅ Scan result: Found {data.get('high_score_count', 0)} stocks")
        
        if data.get('stocks'):
            print("\n📊 Top scoring stocks:")
            for stock in data['stocks'][:3]:  # Show top 3
                print(f"  • {stock['symbol']}: Score {stock['score']}, Price ₹{stock['current_price']}")
    except Exception as e:
        print(f"❌ Scan test failed: {e}")
        return False
    
    # Test stocks list endpoint
    try:
        response = requests.get(f"{base_url}/stocks", timeout=10)
        data = response.json()
        print(f"✅ Stock list: {data.get('total_stocks', 0)} stocks available")
    except Exception as e:
        print(f"❌ Stock list test failed: {e}")
        return False
    
    return True

def main():
    print("🎯 Stock Sniper API - Local Test")
    print("=" * 40)
    
    # Install requirements
    try:
        install_requirements()
    except Exception as e:
        print(f"❌ Failed to install requirements: {e}")
        return
    
    # Start the server
    print("\n🚀 Starting local server...")
    print("Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        # Start server in background for testing
        import uvicorn
        from api.main import app
        
        # Run server
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()
