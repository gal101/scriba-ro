import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, ".venv")
if sys.platform == "win32":
    VENV_PYTHON = os.path.join(VENV_DIR, "Scripts", "python.exe")
    VENV_PYTHONW = os.path.join(VENV_DIR, "Scripts", "pythonw.exe")
else:
    VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python")
    VENV_PYTHONW = os.path.join(VENV_DIR, "bin", "pythonw")

LOG_PATH = os.path.join(BASE_DIR, "scriba.log")

def log_message(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [LAUNCHER] {msg}\n"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except:
        pass

class SplashScreen(tk.Tk):
    def __init__(self):
        super().__init__()
        self.overrideredirect(True)
        self.geometry("400x150")
        self.configure(bg="#1E1E24")
        
        # Center on screen
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"+{x}+{y}")
        
        self.title_label = tk.Label(self, text="Scriba Ro", font=("Segoe UI", 24, "bold"), bg="#1E1E24", fg="#FFFFFF")
        self.title_label.pack(pady=(20, 10))
        
        self.status_label = tk.Label(self, text="Se încarcă...", font=("Segoe UI", 10), bg="#1E1E24", fg="#CCCCCC")
        self.status_label.pack(pady=(0, 20))
        
    def update_status(self, text):
        self.status_label.config(text=text)
        self.update()

def has_nvidia_gpu():
    try:
        if sys.platform == "win32":
            output = subprocess.check_output(
                ["powershell", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                text=True, creationflags=subprocess.CREATE_NO_WINDOW
            )
            return "NVIDIA" in output.upper()
    except Exception as e:
        log_message(f"Eroare detectare GPU: {e}")
    return False

def check_torch_cuda():
    try:
        res = subprocess.run([VENV_PYTHON, "-c", "import torch; exit(0 if torch.cuda.is_available() else 1)"], 
                             capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        return res.returncode == 0
    except:
        return False

def check_torch_installed():
    try:
        res = subprocess.run([VENV_PYTHON, "-c", "import torch; exit(0)"], 
                             capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        return res.returncode == 0
    except:
        return False

def run_setup_logic(splash):
    try:
        log_message("=== PORNIRE LAUNCHER ===")
        # 1. Check and create venv
        if not os.path.exists(VENV_DIR):
            splash.update_status("Se creează mediul virtual...")
            log_message("Creare venv...")
            subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        
        setup_marker = os.path.join(VENV_DIR, ".scriba_setup_complete")
        if not os.path.exists(setup_marker):
            splash.update_status("Se instalează dependențele (poate dura câteva minute)...")
            log_message("Instalare pip upgrade...")
            subprocess.run([VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip"], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            
            is_nvidia = has_nvidia_gpu()
            if is_nvidia:
                log_message("S-a detectat GPU NVIDIA.")
                if not check_torch_cuda():
                    splash.update_status("Se instalează PyTorch cu suport CUDA...")
                    log_message("Instalare PyTorch CUDA...")
                    subprocess.run([VENV_PYTHON, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu121", "--force-reinstall"], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                log_message("Nu s-a detectat GPU NVIDIA.")
                if not check_torch_installed():
                    splash.update_status("Se instalează PyTorch (CPU)...")
                    log_message("Instalare PyTorch CPU...")
                    subprocess.run([VENV_PYTHON, "-m", "pip", "install", "torch"], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
                    
            splash.update_status("Se instalează pachetele necesare...")
            log_message("Instalare requirements.txt...")
            subprocess.run([VENV_PYTHON, "-m", "pip", "install", "-r", "requirements.txt"], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            
            with open(setup_marker, "w") as f:
                f.write("done")
            log_message("Setup complet.")
            
        # 2. Check Models
        models_marker = os.path.join(BASE_DIR, "models", ".scriba_models_installed")
        if not os.path.exists(models_marker):
            log_message("Modelul(ele) lipsește. Se lansează setup.py.")
            splash.destroy()
            subprocess.Popen([VENV_PYTHON, "setup.py"], cwd=BASE_DIR, creationflags=subprocess.CREATE_NO_WINDOW)
            sys.exit(0)
            
        # 3. Launch App
        log_message("Toate pre-rechizitele îndeplinite. Se lansează app.py.")
        splash.update_status("Scriba Ro pornește în fundal.\nVerificați colțul dreapta-jos (lângă ceas)\nîn aproximativ 1-2 minute!")
        subprocess.Popen([VENV_PYTHON, "app.py"], cwd=BASE_DIR, creationflags=subprocess.CREATE_NO_WINDOW)
        time.sleep(5)
        splash.destroy()
        sys.exit(0)
        
    except subprocess.CalledProcessError as e:
        log_message(f"Eroare proces subprocess: {e}")
        messagebox.showerror("Scriba Ro - Eroare", f"Eroare în timpul instalării dependențelor.\nVerificați fisierul scriba.log pentru detalii.")
        splash.destroy()
        sys.exit(1)
    except Exception as e:
        log_message(f"Eroare generală: {e}")
        messagebox.showerror("Scriba Ro - Eroare", f"Eroare internă: {e}")
        splash.destroy()
        sys.exit(1)

if __name__ == "__main__":
    app = SplashScreen()
    threading.Thread(target=run_setup_logic, args=(app,), daemon=True).start()
    app.mainloop()
