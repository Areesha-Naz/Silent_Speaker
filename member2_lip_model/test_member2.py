"""
Test script for Member 2's lip reading model
"""

import sys
import numpy as np
import cv2

# Add src to path
sys.path.append('src')

try:
    # Try to import their model
    from lip_reader import LipReadingModel, predict_from_lips
    print("✅ Successfully imported model")
    
    # Initialize model
    model = LipReadingModel()
    print("✅ Model initialized")
    
    # Create fake test data (16 frames of 128x64)
    print("\nCreating test data...")
    fake_frames = []
    for i in range(16):
        # Random noise image (simulating lip crop)
        fake_frame = np.random.randint(0, 255, (64, 128), dtype=np.uint8)
        fake_frames.append(fake_frame)
    
    # Test prediction
    print("Running prediction...")
    result = predict_from_lips(fake_frames)
    
    if result:
        text, confidence = result
        print(f"\n✅ Prediction successful!")
        print(f"   Text: {text}")
        print(f"   Confidence: {confidence}")
    else:
        print("\n❌ Prediction returned None")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Test complete!")