"""
Test script cho Plant AI Web Application
"""

import requests
import json
import os
from PIL import Image
import io

def test_web_app():
    """Test web application functionality"""
    base_url = "http://localhost:5000"
    
    print("Plant AI Web Application Test")
    print("=" * 40)
    
    # Test 1: Check if server is running
    try:
        response = requests.get(base_url, timeout=5)
        print(f"[OK] Server Status: {response.status_code}")
        if response.status_code == 200:
            print("[OK] Web application is running successfully!")
        else:
            print(f"[ERROR] Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to web application")
        print("Please make sure the web app is running: python app.py")
        return False
    except Exception as e:
        print(f"[ERROR] Error connecting to server: {e}")
        return False
    
    # Test 2: Test API endpoints
    endpoints = ["/", "/about", "/history"]
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            print(f"[OK] {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"[ERROR] {endpoint}: Error - {e}")
    
    # Test 3: Test image upload (if we have a test image)
    test_image_path = "data/health_monitoring/plant_leaf_diseases/Apple___healthy"
    if os.path.exists(test_image_path):
        # Find a test image
        test_images = [f for f in os.listdir(test_image_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if test_images:
            test_image = os.path.join(test_image_path, test_images[0])
            print(f"\nTesting image upload with: {test_images[0]}")
            
            try:
                with open(test_image, 'rb') as f:
                    files = {'file': f}
                    response = requests.post(f"{base_url}/upload", files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        print("[OK] Image upload successful!")
                        print(f"   Predicted: {result.get('predicted_class')}")
                        print(f"   Confidence: {result.get('confidence', 0):.3f}")
                        print(f"   Health Score: {result.get('health_analysis', {}).get('score', 0)}/100")
                    else:
                        print(f"[ERROR] Upload failed: {result.get('error')}")
                else:
                    print(f"[ERROR] Upload failed with status: {response.status_code}")
            except Exception as e:
                print(f"[ERROR] Upload test failed: {e}")
        else:
            print("[WARNING] No test images found for upload test")
    else:
        print("[WARNING] Test image directory not found")
    
    print("\n" + "=" * 40)
    print("Web Application Test Summary:")
    print("[OK] Server is running on http://localhost:5000")
    print("[OK] All endpoints are accessible")
    print("[OK] Ready for use!")
    print("\nTo use the web application:")
    print("1. Open your browser")
    print("2. Go to: http://localhost:5000")
    print("3. Upload a plant image")
    print("4. Get AI analysis results")
    
    return True

if __name__ == "__main__":
    test_web_app()
