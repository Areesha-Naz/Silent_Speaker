"""
Lip Processor for Silent Speaker
Member 1 - For MediaPipe 0.10.32
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path

class LipProcessor:
    def __init__(self):
        """Initialize MediaPipe Face Detection"""
        print("="*60)
        print("LIP PROCESSOR INITIALIZING...")
        print("="*60)
        
        # For MediaPipe 0.10.32, we use tasks.vision
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Create face detector options
            base_options = python.BaseOptions(model_asset_path=None)
            options = vision.FaceDetectorOptions(base_options=base_options)
            self.detector = vision.FaceDetector.create_from_options(options)
            print("✅ Using MediaPipe Tasks Vision")
        except:
            # Fallback to simpler method
            print("⚠️ Using alternative method")
            self.detector = None
        
        # For face mesh (landmarks) - if available in your version
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✅ Face Mesh loaded")
        except:
            print("⚠️ Face Mesh not available, using Haar Cascade")
            self.face_mesh = None
            # Fallback to Haar Cascade for face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Lip landmark indices (standard MediaPipe indices)
        self.lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                            291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
                            78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
        self.lip_indices = list(set(self.lip_indices))
        
        self.target_width = 128
        self.target_height = 64
        self.padding = 20
        
        # Performance
        self.frame_count = 0
        self.fps = 0
        self.fps_timer = time.time()
        
        Path("outputs").mkdir(exist_ok=True)
        
        print(f"✅ Target size: {self.target_width}x{self.target_height}")
        print("✅ Ready!\n")
    
    def get_lip_region(self, frame):
        """Main function - returns lip crop and display frame"""
        display = frame.copy()
        h, w = frame.shape[:2]
        lip_crop = None
        face_detected = False
        bbox = None
        
        # Method 1: Try using face_mesh if available
        if self.face_mesh is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                face_detected = True
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Get lip points
                points = []
                for idx in self.lip_indices:
                    landmark = landmarks[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append([x, y])
                
                if points:
                    points = np.array(points)
                    x1 = max(0, np.min(points[:, 0]) - self.padding)
                    x2 = min(w, np.max(points[:, 0]) + self.padding)
                    y1 = max(0, np.min(points[:, 1]) - self.padding)
                    y2 = min(h, np.max(points[:, 1]) + self.padding)
                    
                    if x2 > x1 and y2 > y1:
                        bbox = (int(x1), int(y1), int(x2), int(y2))
                        cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                        
                        # Crop and process
                        crop = frame[int(y1):int(y2), int(x1):int(x2)]
                        if crop.size > 0:
                            crop = cv2.resize(crop, (self.target_width, self.target_height))
                            lip_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Method 2: Fallback to simple face detection
        if not face_detected and hasattr(self, 'face_cascade'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                face_detected = True
                x, y, fw, fh = faces[0]
                
                # Estimate lip region (lower part of face)
                lip_y = y + int(fh * 0.6)
                lip_h = int(fh * 0.3)
                lip_x = x + int(fw * 0.2)
                lip_w = int(fw * 0.6)
                
                bbox = (lip_x, lip_y, lip_x + lip_w, lip_y + lip_h)
                cv2.rectangle(display, (lip_x, lip_y), (lip_x + lip_w, lip_y + lip_h), (0,255,0), 2)
                
                # Crop and process
                crop = frame[lip_y:lip_y+lip_h, lip_x:lip_x+lip_w]
                if crop.size > 0:
                    crop = cv2.resize(crop, (self.target_width, self.target_height))
                    lip_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # FPS calculation
        self.frame_count += 1
        if time.time() - self.fps_timer >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_timer = time.time()
        
        # Status text
        status = "FACE DETECTED" if face_detected else "NO FACE"
        color = (0,255,0) if face_detected else (0,0,255)
        cv2.putText(display, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(display, f"FPS: {self.fps}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        return lip_crop, display, face_detected, bbox
    
    def release(self):
        """Clean up"""
        if hasattr(self, 'face_mesh') and self.face_mesh:
            self.face_mesh.close()
        print("✅ Resources released")

# Test
if __name__ == "__main__":
    detector = LipProcessor()
    cap = cv2.VideoCapture(0)
    
    print("\n📸 Testing... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        lip, disp, detected, box = detector.get_lip_region(frame)
        cv2.imshow('Lip Detection Test', disp)
        
        if lip is not None:
            cv2.imshow('Lip Crop (128x64)', lip)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()