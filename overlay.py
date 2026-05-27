import tkinter as tk
import threading
import time
import math
import random

class DictationOverlay:
    def __init__(self, parent=None, on_language_toggle=None, initial_lang="ro"):
        self.parent = parent
        self.on_language_toggle = on_language_toggle
        self.initial_lang = initial_lang
        self.root = None
        self.label_text = None
        self.indicator = None
        self.is_running = False
        self.pulse_direction = 1
        self.pulse_alpha = 1.0
        self.state = "idle" # "idle", "listening", "transcribing"
        
        # Audio visualizer variables
        self.current_volume = 0.0
        self.bar_heights = [4.0, 4.0, 4.0, 4.0, 4.0]
        
        # Initialize UI on the thread that instantiated us (main thread)
        self._init_gui()
        
    def _init_gui(self):
        if self.parent:
            self.root = tk.Toplevel(self.parent)
        else:
            self.root = tk.Tk()
            
        self.root.withdraw() # Start hidden
        
        # Make frameless and always on top
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        
        # Windows specific glassmorphism trick: set window alpha transparency
        self.root.attributes("-alpha", 0.92)
        
        # Use a background key color that we will make transparent
        bg_key = "#010101"
        self.root.configure(bg=bg_key)
        self.root.attributes("-transparentcolor", bg_key)
        
        # Main Pill Frame (rounded-like effect using padding and flat borders)
        self.pill_frame = tk.Frame(
            self.root,
            bg="#18181C",
            highlightthickness=1,
            highlightbackground="#33333C",
            padx=15,
            pady=10
        )
        self.pill_frame.pack(padx=10, pady=10)
        
        # 1. Recording status indicator (a colored dot) (left side of pill)
        self.canvas = tk.Canvas(
            self.pill_frame,
            width=16,
            height=16,
            bg="#18181C",
            highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT, padx=(0, 10))
        self.dot = self.canvas.create_oval(2, 2, 14, 14, fill="#FF3B30", width=0)
        
        # 2. Text label (middle of pill)
        self.label = tk.Label(
            self.pill_frame,
            text="Înregistrez...",
            fg="#FFFFFF",
            bg="#18181C",
            font=("Segoe UI", 11, "bold")
        )
        self.label.pack(side=tk.LEFT)
        
        # 3. Audio volume visualizer canvas (5 rounded bars) (right side of pill)
        self.visualizer_canvas = tk.Canvas(
            self.pill_frame,
            width=70,
            height=24,
            bg="#18181C",
            highlightthickness=0
        )
        self.visualizer_canvas.pack(side=tk.LEFT, padx=(10, 0))
        
        self.bars = []
        # Create 5 vertical lines with rounded ends as visualizer bars
        for i in range(5):
            x = 10 + i * 14
            # Draw line with rounded ends as thick line
            bar = self.visualizer_canvas.create_line(
                x, 10, x, 14, 
                width=6, 
                capstyle=tk.ROUND, 
                fill="#55555A"
            )
            self.bars.append(bar)
            

        # 4. Language Toggle Switch (far right of pill)
        self.toggle_frame = tk.Frame(
            self.pill_frame,
            bg="#2A2A30",
            cursor="hand2",
            padx=2,
            pady=2
        )
        self.toggle_frame.pack(side=tk.LEFT, padx=(15, 0))
        
        self.toggle_label = tk.Label(
            self.toggle_frame,
            text="🇷🇴 RO",
            fg="#FFFFFF",
            bg="#2A2A30",
            font=("Segoe UI", 9, "bold"),
            padx=6,
            pady=1
        )
        self.toggle_label.pack()
        
        # Bind click events for language toggle
        self.toggle_frame.bind("<Button-1>", self._on_toggle_click)
        self.toggle_label.bind("<Button-1>", self._on_toggle_click)
        
        # Position at top-center of screen
        self.root.update_idletasks()
        
        # Initialize language button appearance
        self.update_language_ui(self.initial_lang)
        width = self.root.winfo_reqwidth()
        height = self.root.winfo_reqheight()
        
        screen_width = self.root.winfo_screenwidth()
        x = (screen_width // 2) - (width // 2)
        y = 50 # 50 pixels down from top
        
        self.root.geometry(f"+{x}+{y}")
        self.is_running = True
        
        # Start pulsing and visualizer animation
        self._pulse_animation()
        
        # If we are the standalone Tk, run mainloop
        if not self.parent:
            self.root.mainloop()
        
    def _pulse_animation(self):
        if not self.is_running:
            return
            
        try:
            if self.state == "listening":
                # 1. Pulse red dot
                self.pulse_alpha += self.pulse_direction * 0.1
                if self.pulse_alpha >= 1.0:
                    self.pulse_alpha = 1.0
                    self.pulse_direction = -1
                elif self.pulse_alpha <= 0.4:
                    self.pulse_alpha = 0.4
                    self.pulse_direction = 1
                
                r = int(255 * self.pulse_alpha)
                color = f"#{r:02x}3b30"
                self.canvas.itemconfig(self.dot, fill=color)
                
                # 2. Animate and decay the visualizer bars smoothly matching voice input
                for i in range(5):
                    # Outer bars are less sensitive, middle bar is most sensitive
                    sensitivities = [0.4, 0.8, 1.0, 0.8, 0.4]
                    sens = sensitivities[i]
                    
                    # Target height for this bar based on current volume
                    vol = getattr(self, 'current_volume', 0.0)
                    target_h = 4 + (vol * sens * 16)
                    # Add tiny organic flutter if there is active voice
                    if vol > 0.01:
                        target_h += random.uniform(-2, 2)
                    target_h = max(4, min(20, target_h))
                    
                    # Smooth interpolation
                    curr_h = self.bar_heights[i]
                    new_h = curr_h + (target_h - curr_h) * 0.35
                    self.bar_heights[i] = new_h
                    
                    # Update coordinates on canvas
                    x = 10 + i * 14
                    y0 = 12 - new_h / 2
                    y1 = 12 + new_h / 2
                    self.visualizer_canvas.coords(self.bars[i], x, y0, x, y1)
                    self.visualizer_canvas.itemconfig(self.bars[i], fill="#FF3B30") # Vibrant red bars
                    
                # Smoothly decay volume to simulate real-time drop when silent
                self.current_volume *= 0.8
                
            elif self.state == "transcribing":
                # Pulse orange dot faster
                self.pulse_alpha += self.pulse_direction * 0.15
                if self.pulse_alpha >= 1.0:
                    self.pulse_alpha = 1.0
                    self.pulse_direction = -1
                elif self.pulse_alpha <= 0.3:
                    self.pulse_alpha = 0.3
                    self.pulse_direction = 1
                
                r = int(255 * self.pulse_alpha)
                g = int(149 * self.pulse_alpha)
                color = f"#{r:02x}{g:02x}00"
                self.canvas.itemconfig(self.dot, fill=color)
                
                # Wave loading animation in warm orange
                t = time.time() * 12
                for i in range(5):
                    new_h = 4 + (abs(math.sin(t + i * 0.8)) * 12)
                    self.bar_heights[i] = new_h
                    x = 10 + i * 14
                    y0 = 12 - new_h / 2
                    y1 = 12 + new_h / 2
                    self.visualizer_canvas.coords(self.bars[i], x, y0, x, y1)
                    self.visualizer_canvas.itemconfig(self.bars[i], fill="#FF9500") # Loading orange
                    
            else:
                # Idle state: keep bars flat and dark grey
                for i in range(5):
                    x = 10 + i * 14
                    self.visualizer_canvas.coords(self.bars[i], x, 10, x, 14)
                    self.visualizer_canvas.itemconfig(self.bars[i], fill="#55555A") # Standby grey
                    
        except Exception:
            pass
            
        # Run at 40ms interval (25 FPS) for ultra-fluid motion physics
        self.root.after(40, self._pulse_animation)
        
    def update_volume(self, volume):
        """Thread-safe update of current volume level (0.0 to 1.0)"""
        self.current_volume = volume
        
    def show_listening(self, lang="ro"):
        """Changes state to listening and displays overlay."""
        if not self.root:
            return
            
        def _update():
            self.state = "listening"
            text = "Scriba: Ascult..." if lang == "ro" else "Scriba: Listening..."
            self.label.configure(text=text, fg="#FF3B30")
            self.canvas.itemconfig(self.dot, fill="#FF3B30")
            
            # Position correctly (re-center based on length)
            self.root.update_idletasks()
            width = self.root.winfo_reqwidth()
            screen_width = self.root.winfo_screenwidth()
            x = (screen_width // 2) - (width // 2)
            self.root.geometry(f"+{x}+50")
            
            self.root.deiconify()
            self.root.lift()
            
        self.root.after(0, _update)
        
    def show_transcribing(self, lang="ro"):
        """Changes state to transcribing."""
        if not self.root:
            return
            
        def _update():
            self.state = "transcribing"
            text = "Scriba: Transcriu..." if lang == "ro" else "Scriba: Transcribing..."
            self.label.configure(text=text, fg="#FF9500")
            self.canvas.itemconfig(self.dot, fill="#FF9500")
            
            # Position correctly (re-center based on length)
            self.root.update_idletasks()
            width = self.root.winfo_reqwidth()
            screen_width = self.root.winfo_screenwidth()
            x = (screen_width // 2) - (width // 2)
            self.root.geometry(f"+{x}+50")
            
        self.root.after(0, _update)
        
    def hide(self):
        """Hides the overlay."""
        if not self.root:
            return
            
        def _update():
            self.state = "idle"
            self.root.withdraw()
            
        self.root.after(0, _update)
        
    def _on_toggle_click(self, event=None):
        if self.on_language_toggle:
            self.on_language_toggle()
            
    def update_language_ui(self, lang):
        """Updates the language toggle button appearance based on the active language."""
        if not self.root:
            return
            
        def _update():
            if lang == "ro":
                self.toggle_frame.configure(bg="#2E7D32") # Dark green
                self.toggle_label.configure(text="🇷🇴 RO", bg="#2E7D32", fg="#FFFFFF")
                if self.state == "listening":
                    self.label.configure(text="Scriba: Ascult...", fg="#FF3B30")
            else:
                self.toggle_frame.configure(bg="#1565C0") # Dark blue
                self.toggle_label.configure(text="🇬🇧 EN", bg="#1565C0", fg="#FFFFFF")
                if self.state == "listening":
                    self.label.configure(text="Scriba: Listening...", fg="#FF3B30")
                    
            # Re-center the overlay on screen since width might have changed
            self.root.update_idletasks()
            width = self.root.winfo_reqwidth()
            screen_width = self.root.winfo_screenwidth()
            x = (screen_width // 2) - (width // 2)
            self.root.geometry(f"+{x}+50")
            
        try:
            self.root.after(0, _update)
        except Exception:
            pass

    def close(self):
        """Closes the overlay GUI."""
        self.is_running = False
        if self.root:
            self.root.after(0, self.root.destroy)

# Self-test code
if __name__ == "__main__":
    overlay = DictationOverlay()
    time.sleep(1)
    print("Showing listening...")
    overlay.show_listening()
    
    # Simulate speak volume updates
    for i in range(50):
        time.sleep(0.1)
        vol = (math.sin(i * 0.4) + 1.0) / 2.0
        overlay.update_volume(vol)
        
    print("Showing transcribing...")
    overlay.show_transcribing()
    time.sleep(3)
    print("Hiding...")
    overlay.hide()
    time.sleep(2)
    overlay.close()
