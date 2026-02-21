"""
Simple demo for Member 3 to understand how to use LipProcessor
"""

import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lip_processor import LipProcessor

def main():
    print("\n" + "="*60)
    print("SIMPLE DEMO - How Member 3 will use this")
    print("="*60)
    
    # Initialize (Member 3 will do this once)
    detector = LipProcessor()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n📋 USAGE EXAMPLE:")
    print("""
    # In Member 3's pipeline:
    
    from src.lip_processor import LipProcessor
    
    # Initialize
    detector = LipProcessor()
    
    # In main loop:
    while True:
        frame = webcam.read()
        
        # THIS IS THE MAIN FUNCTION
        lip_crop, display_frame, face_detected, bbox = detector.get_lip_region(frame)
        
        if face_detected and lip_crop is not None:
            # lip_crop is ready for Member 2's model!
            # Shape: (64, 128) grayscale
            # Pass to buffer for model
            pass
        
        # display_frame has green box - send to Member 4's UI
        # bbox contains coordinates if needed
    """)
    
    print("\n🔴 LIVE DEMO - Press 'q' to quit")
    print("-" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # THE MAIN FUNCTION - exactly what Member 3 will use
        lip_crop, display_frame, face_detected, bbox = detector.get_lip_region(frame)
        
        # Show the display frame (with green box)
        cv2.imshow('Demo - Display Frame (for UI)', display_frame)
        
        # Show lip crop if available
        if lip_crop is not None:
            cv2.imshow('Demo - Lip Crop (for Model)', lip_crop)
            print(f"✅ Lip crop ready! Shape: {lip_crop.shape}")
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE")
    print("📤 Ready to hand off to Member 3!")
    print("="*60)

if __name__ == "__main__":
    main()