import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk
import threading
import time
import sys
import os

# Paths setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.lip_processor import LipProcessor  # Member 1
    from lip_reader import LipReadingModel      # Member 2
    from app_interface import SilentSpeakerApp  # Member 4
except ImportError as e:
    print(f"❌ Error: Files missing! {e}")

class SilentSpeakerMain:
    def __init__(self):
        print("🚀 Starting Final Delivery Version...")
        self.app = SilentSpeakerApp()
        
        # Stop Member 4's default camera and loops immediately
        if hasattr(self.app, 'cap'):
            self.app.cap.release() 
        
        # Start our own camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Placeholders
        self.detector = LipProcessor()
        self.reader = LipReadingModel()
        
        self.latest_frame = None
        self.current_word = "SYSTEM READY"
        self.running = True
        self.is_predicting = False
        
        self.app.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Thread 1: Camera & Vision Processing
        threading.Thread(target=self.camera_thread, daemon=True).start()
        
        # Start Display Loop
        self.update_display()

    def camera_thread(self):
        """Camera reading and AI processing in background"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Member 1: Lip Detection
                lip_crop, display_frame, face_detected, _ = self.detector.get_lip_region(frame)
                
                if face_detected and lip_crop is not None:
                    self.reader.add_frame(lip_crop)
                    
                    # Member 2: AI Prediction (if buffer full)
                    if len(self.reader.buffer) >= 16 and not self.is_predicting:
                        threading.Thread(target=self.predict_async, daemon=True).start()
                
                self.latest_frame = display_frame
            time.sleep(0.01)

    def predict_async(self):
        self.is_predicting = True
        try:
            res = self.reader.predict()
            if res["is_valid"]:
                self.current_word = res["text"].upper()
        except: pass
        self.is_predicting = False

    def update_display(self):
        """Pure UI refresh (No heavy logic)"""
        if self.latest_frame is not None:
            # Conversion to RGB for UI
            img = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(640, 480))
            
            try:
                # Member 4's UI Labels
                self.app.video_label.configure(image=img_ctk)
                self.app.output_text.configure(text=self.current_word)
            except: pass

        if self.running:
            self.app.after(15, self.update_display)

    def on_closing(self):
        self.running = False
        self.cap.release()
        self.app.destroy()
        sys.exit()

if __name__ == "__main__":
    project = SilentSpeakerMain()
    # Force cancel any lingering UI updates from Member 4
    try:
        project.app.after_cancel(project.app.after(10, project.app.update_ui))
    except: pass
    project.app.mainloop()