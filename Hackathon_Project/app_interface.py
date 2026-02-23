import customtkinter as ctk
import cv2
from PIL import Image
import datetime

COLORS = {
    "bg": "#0D1117",
    "card": "#161B22",
    "accent": "#2EB67D",
    "text": "#F0F6FC",
    "ghost": "#8B949E",
    "alert": "#F85149"
}

class SilentSpeakerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Silent Speaker | Assistive VSR Terminal")
        self.geometry("1100x750")
        self.configure(fg_color=COLORS["bg"])

        self.is_recording = False
        self.large_text_mode = False
        self.history_data = []

        # NOTE: Camera is NOT opened here — main.py owns the camera
        self.cap = None

        self.grid_columnconfigure(1, weight=3) 
        self.grid_columnconfigure(0, weight=1) 
        self.grid_rowconfigure(0, weight=1)

        self.setup_sidebar()
        self.setup_main_view()
        # NOTE: update_ui() is NOT called here — main.py handles display

    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="📜 HISTORY LOG", 
                      font=("Inter", 16, "bold"), text_color=COLORS["accent"]).pack(pady=(20, 10))
        
        self.history_log = ctk.CTkTextbox(self.sidebar, fg_color="transparent", 
                                          font=("Inter", 13), text_color=COLORS["text"])
        self.history_log.pack(fill="both", expand=True, padx=15, pady=10)
        self.history_log.configure(state="disabled")

        # --- DELETE HISTORY BUTTONS ---
        self.btn_del_last = ctk.CTkButton(self.sidebar, text="Delete Last Entry", 
                                           fg_color="#343A40", hover_color="#454D55", 
                                           height=30, command=self.delete_last_history)
        self.btn_del_last.pack(pady=5, padx=20, fill="x")

        self.btn_clear_all = ctk.CTkButton(self.sidebar, text="Clear All History", 
                                            fg_color="transparent", text_color=COLORS["alert"], 
                                            hover_color="#2A1215", height=30, command=self.clear_all_history)
        self.btn_clear_all.pack(pady=(0, 20), padx=20, fill="x")

        self.btn_large_text = ctk.CTkCheckBox(self.sidebar, text="Large Text Mode", 
                                               command=self.toggle_accessibility)
        self.btn_large_text.pack(pady=10)

        self.status_dot = ctk.CTkLabel(self.sidebar, text="● SYSTEM READY", 
                                        text_color=COLORS["accent"], font=("Inter", 12, "bold"))
        self.status_dot.pack(side="bottom", pady=20)

    def setup_main_view(self):
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")

        self.video_card = ctk.CTkFrame(self.main_container, fg_color="black", corner_radius=15)
        self.video_card.pack(fill="both", expand=True, pady=(0, 10))
        
        self.video_label = ctk.CTkLabel(self.video_card, text="")
        self.video_label.pack(fill="both", expand=True, padx=5, pady=5)

        self.confidence_bar = ctk.CTkProgressBar(self.main_container, progress_color=COLORS["accent"], height=8)
        self.confidence_bar.pack(fill="x", pady=(5, 10))
        self.confidence_bar.set(0.0)

        self.trans_frame = ctk.CTkFrame(self.main_container, fg_color=COLORS["card"], corner_radius=15)
        self.trans_frame.pack(fill="x", pady=5)
        
        self.output_text = ctk.CTkLabel(self.trans_frame, text="Press Start to Begin...", 
                                         font=("Inter", 24, "bold"), wraplength=600)
        self.output_text.pack(pady=20)

        self.btn_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.btn_frame.pack(fill="x", pady=10)

        self.btn_start = ctk.CTkButton(self.btn_frame, text="START SESSION", fg_color=COLORS["accent"], 
                                        text_color="black", font=("Inter", 14, "bold"), height=40, command=self.toggle_session)
        self.btn_start.pack(side="left", padx=5, expand=True, fill="x")

        self.btn_clear = ctk.CTkButton(self.btn_frame, text="CLEAR SCREEN", fg_color="#343A40", height=40, command=self.clear_screen)
        self.btn_clear.pack(side="left", padx=5, expand=True, fill="x")

        self.btn_save = ctk.CTkButton(self.btn_frame, text="SAVE TO LOG", fg_color="#343A40", height=40, command=self.save_to_history)
        self.btn_save.pack(side="left", padx=5, expand=True, fill="x")

    def toggle_session(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.btn_start.configure(text="STOP SESSION", fg_color=COLORS["alert"])
            self.status_dot.configure(text="● AI LISTENING...", text_color=COLORS["accent"])
            # Show "Listening..." instead of hardcoded prediction
            self.output_text.configure(text="Listening...", text_color=COLORS["text"])
            # Don't set fake confidence — let real predictions update it
        else:
            self.btn_start.configure(text="START SESSION", fg_color=COLORS["accent"])
            self.status_dot.configure(text="● SYSTEM PAUSED", text_color=COLORS["ghost"])
            self.output_text.configure(text="Session Paused", text_color=COLORS["ghost"])
            self.confidence_bar.set(0.0)

    def update_confidence(self, value: float):
        """Update the confidence bar with a real value from predictions"""
        self.confidence_bar.set(min(1.0, max(0.0, value)))

    def update_status(self, text: str, color: str = None):
        """Update the status dot text and color"""
        if color is None:
            color = COLORS["accent"]
        self.status_dot.configure(text=f"● {text}", text_color=color)

    def clear_screen(self):
        self.output_text.configure(text="Waiting...", text_color=COLORS["ghost"])
        self.confidence_bar.set(0)

    def save_to_history(self):
        txt = self.output_text.cget("text")
        if txt not in ("Press Start to Begin...", "Waiting...", "Listening...", "Session Paused", ""):
            ts = datetime.datetime.now().strftime("%H:%M")
            entry = f"[{ts}] {txt}"
            self.history_data.append(entry)
            self.refresh_history_display()

    def delete_last_history(self):
        if self.history_data:
            self.history_data.pop()
            self.refresh_history_display()

    def clear_all_history(self):
        self.history_data = []
        self.refresh_history_display()

    def refresh_history_display(self):
        """Helper to redraw the history sidebar whenever data changes."""
        self.history_log.configure(state="normal")
        self.history_log.delete("1.0", "end")
        for entry in self.history_data:
            self.history_log.insert("end", entry + "\n")
        self.history_log.configure(state="disabled")
        self.history_log.see("end")

    def toggle_accessibility(self):
        self.large_text_mode = not self.large_text_mode
        size = 40 if self.large_text_mode else 24
        self.output_text.configure(font=("Inter", size, "bold"))

if __name__ == "__main__":
    app = SilentSpeakerApp()
    app.mainloop()