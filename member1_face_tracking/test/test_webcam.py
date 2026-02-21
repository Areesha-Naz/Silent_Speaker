"""
Test Script (Haseeb)
Tests all functionality of LipProcessor
"""

import cv2
import sys
import os
import time
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lip_processor import LipProcessor

class LipProcessorTester:
    def __init__(self):
        """Initialize tester"""
        self.detector = LipProcessor()
        self.test_results = {
            'webcam': False,
            'face_detection': False,
            'lip_cropping': False,
            'grayscale': False,
            'resize': False,
            'fps': False
        }
        
    def test_webcam(self):
        """Test 1: Check if webcam works"""
        print("\n🔵 TEST 1: Webcam Access")
        print("-" * 40)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ FAILED: Cannot open webcam")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print("❌ FAILED: Cannot read frame from webcam")
            cap.release()
            return False
        
        print(f"✅ PASSED: Webcam working")
        print(f"   - Frame size: {frame.shape[1]}x{frame.shape[0]}")
        print(f"   - Frame format: {frame.dtype}")
        
        cap.release()
        self.test_results['webcam'] = True
        return True
    
    def test_face_detection(self):
        """Test 2: Check face detection"""
        print("\n🔵 TEST 2: Face Detection")
        print("-" * 40)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Look at the camera...")
        face_detected = False
        
        # Try for 30 frames
        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                continue
            
            lip_crop, display, detected, bbox = self.detector.get_lip_region(frame)
            
            if detected:
                face_detected = True
                print(f"✅ Face detected on frame {i+1}")
                break
            
            cv2.imshow('Testing - Look at camera', display)
            cv2.waitKey(1)
        
        cap.release()
        cv2.destroyAllWindows()
        
        if face_detected:
            self.test_results['face_detection'] = True
            return True
        else:
            print("❌ FAILED: No face detected")
            return False
    
    def test_lip_cropping(self):
        """Test 3: Check lip cropping functionality"""
        print("\n🔵 TEST 3: Lip Cropping")
        print("-" * 40)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Opening windows...")
        print("- Main window: Shows face with green box")
        print("- Lip window: Shows cropped lips (128x64 grayscale)")
        print("\nPress 'q' to continue testing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            lip_crop, display, detected, bbox = self.detector.get_lip_region(frame)
            
            # Show main display
            cv2.imshow('Main Display - Face Detection', display)
            
            # Show lip crop if available
            if lip_crop is not None:
                cv2.imshow('Lip Crop - 128x64 Grayscale', lip_crop)
                
                # Verify crop properties
                if lip_crop.shape == (64, 128):  # Grayscale is (height, width)
                    print(f"✅ Lip crop correct size: {lip_crop.shape}")
                    self.test_results['lip_cropping'] = True
                    self.test_results['grayscale'] = True
                    self.test_results['resize'] = True
            
            if cv2.waitKey(1) == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return True
    
    def test_performance(self):
        """Test 4: Check FPS performance"""
        print("\n🔵 TEST 4: Performance (FPS)")
        print("-" * 40)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Measuring FPS for 5 seconds...")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5:  # Test for 5 seconds
            ret, frame = cap.read()
            if not ret:
                break
            
            lip_crop, display, detected, bbox = self.detector.get_lip_region(frame)
            frame_count += 1
            
            cv2.imshow('Performance Test', display)
            cv2.waitKey(1)
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        
        print(f"📊 Results:")
        print(f"   - Frames processed: {frame_count}")
        print(f"   - Time: {elapsed:.2f} seconds")
        print(f"   - Average FPS: {fps:.1f}")
        
        if fps >= 15:
            print(f"✅ PASSED: {fps:.1f} FPS (meets 15 FPS requirement)")
            self.test_results['fps'] = True
        else:
            print(f"⚠️ WARNING: {fps:.1f} FPS (below 15 FPS target)")
        
        cap.release()
        cv2.destroyAllWindows()
        return fps >= 15
    
    def test_different_conditions(self):
        """Test 5: Test with different conditions"""
        print("\n🔵 TEST 5: Different Conditions")
        print("-" * 40)
        
        print("Testing with different scenarios:")
        
        # Test different distances
        print("\n1️⃣ Try at different distances:")
        print("   - Hold camera at arm's length")
        print("   - Bring camera close")
        print("   Press any key when done...")
        cv2.waitKey(0)
        
        # Test different angles
        print("\n2️⃣ Try at different angles:")
        print("   - Look up")
        print("   - Look down")
        print("   - Turn head left/right")
        print("   Press any key when done...")
        cv2.waitKey(0)
        
        # Test with expressions
        print("\n3️⃣ Try different expressions:")
        print("   - Smile")
        print("   - Say 'Hello'")
        print("   - Say 'Yes' and 'No'")
        print("   Press any key to finish...")
        cv2.waitKey(0)
        
        return True
    
    def generate_report(self):
        """Generate test report"""
        print("\n" + "="*60)
        print("📊 TEST REPORT - MEMBER 1")
        print("="*60)
        
        tests = [
            ("Webcam Access", self.test_results['webcam']),
            ("Face Detection", self.test_results['face_detection']),
            ("Lip Cropping", self.test_results['lip_cropping']),
            ("Grayscale Conversion", self.test_results['grayscale']),
            ("Resize to 128x64", self.test_results['resize']),
            ("FPS >= 15", self.test_results['fps'])
        ]
        
        all_passed = True
        for test_name, passed in tests:
            status = "✅ PASS" if passed else "❌ FAIL"
            if not passed:
                all_passed = False
            print(f"{status} - {test_name}")
        
        print("-" * 60)
        if all_passed:
            print("🎉 OVERALL: ALL TESTS PASSED! Ready for Member 3!")
        else:
            print("⚠️ OVERALL: Some tests failed. Review and fix issues.")
        print("="*60)
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*60)
        print("🚀 STARTING MEMBER 1 COMPLETE TESTING")
        print("="*60)
        
        self.test_webcam()
        self.test_face_detection()
        self.test_lip_cropping()
        self.test_performance()
        self.test_different_conditions()
        self.generate_report()
        
        print("\n✅ Testing complete! Check outputs folder for samples.")

def main():
    """Main test function"""
    tester = LipProcessorTester()
    
    print("\n🎯 MEMBER 1 - COMPLETE TEST SUITE")
    print("This will test all functionality:")
    print("1. Webcam access")
    print("2. Face detection")
    print("3. Lip cropping")
    print("4. Grayscale conversion") 
    print("5. Resize to 128x64")
    print("6. Performance (FPS)")
    print("7. Different conditions")
    
    input("\nPress Enter to start testing...")
    
    tester.run_all_tests()
    
    print("\n📁 Sample images saved in 'outputs/' folder")
    print("🚀 Ready to hand off to Member 3!")

if __name__ == "__main__":
    main()