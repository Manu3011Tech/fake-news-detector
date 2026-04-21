#!/usr/bin/env python3
"""
Run the Fake News Detection System - FIXED VERSION
"""

import os
import sys
import subprocess
import webbrowser
import time
import requests

def check_server(url, max_attempts=10):
    """Wait for server to start"""
    print("⏳ Waiting for server to start...", end="", flush=True)
    for i in range(max_attempts):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(" ✅")
                return True
        except:
            print(".", end="", flush=True)
            time.sleep(1)
    print(" ❌")
    return False

def start_server():
    """Start the FastAPI server"""
    print("\n" + "="*50)
    print("🚀 Starting Fake News Detection Server")
    print("="*50)
    
    # Check if backend directory exists
    if not os.path.exists('backend'):
        print("❌ Error: 'backend' directory not found!")
        return
    
    # Change to backend directory
    backend_dir = os.path.join(os.getcwd(), 'backend')
    os.chdir(backend_dir)
    
    # Start the server
    print("\n📡 Starting server...")
    
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", 
         "--host", "127.0.0.1", "--port", "8000", "--reload"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(3)
    
    # Check if server is running
    if check_server("http://127.0.0.1:8000/health"):
        print("\n✨ Server is running!")
        print("📍 API URL: http://127.0.0.1:8000")
        print("📍 API Docs: http://127.0.0.1:8000/docs")
        
        # Try different URLs
        urls_to_try = [
            "http://127.0.0.1:8000/",
            "http://127.0.0.1:8000/static/index.html",
            "http://127.0.0.1:8000/static/simple.html"
        ]
        
        print("\n🌐 Opening frontend...")
        for url in urls_to_try:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    webbrowser.open(url)
                    print(f"   ✅ Opened: {url}")
                    break
            except:
                continue
        else:
            print("   ℹ️ Open manually: http://127.0.0.1:8000/docs")
    else:
        print("\n❌ Server failed to start!")
        print("   Check for errors above")
    
    print("\n⚠️ Press Ctrl+C to stop the server\n")
    
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        server_process.terminate()
        server_process.wait()
        print("✅ Server stopped!")

def main():
    """Main function"""
    print("="*50)
    print("🔍 Fake News Detection System - FIXED")
    print("="*50)
    
    # Change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"📁 Working directory: {script_dir}")
    
    # Install requests if not present
    try:
        import requests
    except ImportError:
        print("📦 Installing requests...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()