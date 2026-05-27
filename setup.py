import os
import sys

# Guarantee that the application's directory is always in Python's search path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import shutil
import threading
import time

def check_and_install_dependencies():
    """Double checks that we are running inside the venv and requirements are met."""
    try:
        import customtkinter
        import torch
        import faster_whisper
        import pynput
        import pyperclip
        import pystray
        import PIL
    except ImportError:
        print("Dependencies missing, installing from requirements.txt...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

# Run dependency check first
check_and_install_dependencies()

import customtkinter as ctk
import torch
from huggingface_hub import snapshot_download

# Define constants
MODEL_ID = "openai/whisper-large-v3-turbo"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RAW_MODEL_DIR = os.path.join(MODELS_DIR, "raw-whisper-large-v3-turbo")
CT2_MODEL_DIR = os.path.join(MODELS_DIR, "whisper-large-v3-turbo-ct2")
LOG_PATH = os.path.join(BASE_DIR, "scriba.log")

def log_message(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [SETUP] {message}\n"
    try:
        # Safely print to Windows console even if console code page does not support unicode characters
        print(message.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
    except Exception:
        try:
            print(message)
        except Exception:
            pass
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception:
        pass

class SetupGUI(ctk.CTk):
    def __init__(self, cuda_available):
        super().__init__()
        
        self.cuda_available = cuda_available
        self.quantization = "float16" if cuda_available else "int8"
        self.device_name = "NVIDIA GPU (CUDA - viteză maximă)" if cuda_available else "CPU (procesor - optimizat INT8)"
        
        # Configure window
        self.title("Scriba Ro - Configurare Asistent Dictare")
        self.geometry("600x420")
        self.resizable(False, False)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # App Icon placeholder/aesthetic
        self.icon_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.icon_frame.pack(pady=(20, 5))
        
        self.mic_label = ctk.CTkLabel(
            self.icon_frame, 
            text="🎙️", 
            font=ctk.CTkFont(size=50)
        )
        self.mic_label.pack()
        
        self.title_label = ctk.CTkLabel(
            self, 
            text="Scriba Ro", 
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold")
        )
        self.title_label.pack(pady=5)
        
        self.subtitle_label = ctk.CTkLabel(
            self, 
            text="Asistentul tău local și offline de dictare în limba română", 
            font=ctk.CTkFont(family="Segoe UI", size=14, slant="italic"),
            text_color="#888888"
        )
        self.subtitle_label.pack(pady=(0, 20))
        
        # Details card
        self.card = ctk.CTkFrame(self, fg_color="#1E1E24", border_width=1, border_color="#33333C")
        self.card.pack(fill="x", padx=40, pady=10)
        
        self.hardware_label = ctk.CTkLabel(
            self.card, 
            text=f"Hardware detectat: {self.device_name}", 
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#4CAF50" if cuda_available else "#FF9800"
        )
        self.hardware_label.pack(pady=(15, 5), padx=20, anchor="w")
        
        self.info_text = (
            "Aplicația va descărca modelul oficial 'openai/whisper-large-v3-turbo' (~3.09 GB)\n"
            "Cel mai bun model multilingual OpenAI — română, engleză și alte 97 de limbi.\n"
            "Modelul va fi optimizat pentru calculatorul tău (conversie CTranslate2, ~1.6 GB final).\n"
            "Acest proces se rulează o singură dată și necesită conexiune la internet."
        )
        self.info_label = ctk.CTkLabel(
            self.card, 
            text=self.info_text, 
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color="#CCCCCC",
            justify="left"
        )
        self.info_label.pack(pady=(0, 15), padx=20, anchor="w")
        
        # Progress element (hidden initially)
        self.progress_bar = ctk.CTkProgressBar(self, width=520)
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(
            self, 
            text="", 
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
        )
        
        # Action button
        self.action_button = ctk.CTkButton(
            self, 
            text="Descarcă și Instalează Modelul", 
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            height=40,
            command=self.start_setup_thread
        )
        self.action_button.pack(pady=20)
        
    def start_setup_thread(self):
        self.action_button.configure(state="disabled", text="Se instalează...")
        self.progress_bar.pack(pady=10)
        self.status_label.pack(pady=5)
        
        # Start background thread
        threading.Thread(target=self.run_setup, daemon=True).start()
        
    def monitor_download(self):
        total_expected_bytes = 3.09 * 1024 * 1024 * 1024 # ~3.09 GB (float32 safetensors)
        while not self.download_finished and hasattr(self, 'is_running') and self.winfo_exists():
            size_bytes = 0
            if os.path.exists(RAW_MODEL_DIR):
                for root, dirs, files in os.walk(RAW_MODEL_DIR):
                    # Skip cache or checkpoints if any slipped in
                    if "checkpoint" in root:
                        continue
                    for f in files:
                        fp = os.path.join(root, f)
                        try:
                            if os.path.exists(fp):
                                size_bytes += os.path.getsize(fp)
                        except Exception:
                            pass
            
            if size_bytes > 0:
                progress = min(size_bytes / total_expected_bytes, 0.99)
                pct = int(progress * 100)
                gb_downloaded = size_bytes / (1024 * 1024 * 1024)
                
                # Update GUI thread-safely
                def _update_gui(p=progress, g=gb_downloaded, c=pct):
                    self.status_label.configure(
                        text=f"Etapa 1/3: Se descarcă modelul... {g:.2f} GB / 3.09 GB ({c}%)",
                        text_color="#FFFFFF"
                    )
                    self.progress_bar.set(p * 0.4) # Map download to 0% - 40% of progress bar
                
                try:
                    self.after(0, _update_gui)
                except Exception:
                    break
            time.sleep(0.5)
        
    def run_setup(self):
        try:
            log_message("Începe configurarea modelului local...")
            
            # Pre-clean previous partial/corrupted downloads (which may contain heavy checkpoints) to start fresh and fast
            if os.path.exists(RAW_MODEL_DIR):
                log_message("Curățare descărcare parțială anterioară (inclusiv checkpoint-uri inutile)...")
                shutil.rmtree(RAW_MODEL_DIR)
                
            os.makedirs(MODELS_DIR, exist_ok=True)
            
            # Step 1: Download model from HF
            log_message(f"Pasul 1: Se inițiază descărcarea modelului {MODEL_ID} de pe Hugging Face...")
            self.update_status("Etapa 1/3: Se descarcă modelul din Hugging Face (aprox. 3.09 GB)...")
            self.progress_bar.configure(mode="determinate")
            self.progress_bar.set(0)
            
            # Start parallel download monitor thread
            self.download_finished = False
            self.is_running = True
            threading.Thread(target=self.monitor_download, daemon=True).start()
            
            # Download ONLY the core model files, ignoring the huge redundant checkpoint folders!
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=RAW_MODEL_DIR,
                local_dir_use_symlinks=False,
                ignore_patterns=["checkpoint-*", "checkpoint-*/*"]
            )
            
            self.download_finished = True
            log_message("Descărcare model din Hugging Face finalizată cu succes.")
            self.progress_bar.set(0.4)
            
            # Step 2: Convert model to CTranslate2
            log_message(f"Pasul 2: Conversie model în format CTranslate2 ({self.quantization})...")
            self.update_status("Etapa 2/3: Se optimizează modelul local (conversie CTranslate2)...")
            self.progress_bar.set(0.5)
            
            import subprocess
            
            # Dynamically determine which potential files actually exist to copy them safely without crashing the converter
            candidate_files = [
                "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", 
                "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json", "normalizer.json"
            ]
            copy_files = [f for f in candidate_files if os.path.exists(os.path.join(RAW_MODEL_DIR, f))]
            log_message(f"Fișiere detectate pentru copiere în modelul convertit: {copy_files}")
            
            # Get the path to ct2-transformers-converter in the current environment
            converter_path = os.path.join(sys.prefix, "Scripts", "ct2-transformers-converter.exe")
            if not os.path.exists(converter_path):
                # Fallback to python module execution if executable is not direct
                cmd = [
                    sys.executable, "-m", "ctranslate2.converters.transformers",
                    "--model", RAW_MODEL_DIR,
                    "--output_dir", CT2_MODEL_DIR,
                    "--quantization", self.quantization,
                ]
            else:
                cmd = [
                    converter_path,
                    "--model", RAW_MODEL_DIR,
                    "--output_dir", CT2_MODEL_DIR,
                    "--quantization", self.quantization,
                ]
            
            if copy_files:
                cmd.append("--copy_files")
                cmd.extend(copy_files)
                
            log_message(f"Rulare comandă conversie: {' '.join(cmd)}")
            # Run conversion
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            log_message("Conversie finalizată cu succes.")
            
            self.progress_bar.set(0.8)
            
            # Step 3: Cleanup temporary raw model files to save space
            log_message("Pasul 3: Curățare fișiere brute temporare din models/raw-whisper-large-v3-turbo...")
            self.update_status("Etapa 3/3: Curățare fișiere temporare...")
            if os.path.exists(RAW_MODEL_DIR):
                shutil.rmtree(RAW_MODEL_DIR)
            log_message("Fișiere temporare șterse. Spațiu economisit.")
                
            self.progress_bar.set(1.0)
            self.update_status("Configurare finalizată cu succes!", success=True)
            log_message("Pregătire Scriba Ro încheiată cu succes! Modelul este optimizat și gata de utilizare.")
            
            # Update GUI when done
            self.after(0, self.setup_completed)
            
        except subprocess.CalledProcessError as e:
            log_message(f"CRITICAL ERROR: Conversia CTranslate2 a eșuat cu codul {e.returncode}!")
            log_message(f"Eroarea raportată (stderr):\n{e.stderr}")
            log_message(f"Output-ul standard (stdout):\n{e.stdout}")
            self.progress_bar.stop()
            self.update_status("Eroare conversie model local!", error=True)
            self.after(0, lambda: self.action_button.configure(state="normal", text="Reîncearcă Instalarea"))
            
        except Exception as e:
            log_message(f"CRITICAL ERROR în timpul configurării: {str(e)}")
            self.progress_bar.stop()
            self.update_status(f"Eroare: {str(e)}", error=True)
            self.after(0, lambda: self.action_button.configure(state="normal", text="Reîncearcă Instalarea"))
            
    def update_status(self, text, error=False, success=False):
        color = "#FF5252" if error else ("#4CAF50" if success else "#FFFFFF")
        self.after(0, lambda: self.status_label.configure(text=text, text_color=color))
        
    def setup_completed(self):
        self.action_button.configure(
            state="normal", 
            text="Lansează Scriba Ro", 
            fg_color="#4CAF50", 
            hover_color="#45a049",
            command=self.launch_app
        )
        
    def launch_app(self):
        self.destroy()
        # Launch main app script
        import subprocess
        app_script = os.path.join(BASE_DIR, "app.py")
        # Run app.py with pythonw.exe to natively register as a Windows GUI subsystem application
        pythonw_exe = sys.executable.replace("python.exe", "pythonw.exe")
        subprocess.Popen([pythonw_exe, app_script], cwd=BASE_DIR)
        sys.exit(0)

def main():
    # Clear log file on a fresh launcher run to keep it clean and readable for only the current run
    try:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass

    log_message("==========================================")
    log_message("     PORNIRE SETUP / LAUNCHER SCRIBA RO")
    log_message("==========================================")
    log_message(f"Executabil Python: {sys.executable}")
    
    # Check PyTorch version and CUDA support
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        cuda_version = torch.version.cuda
        
        log_message(f"Versiune PyTorch instalată: {torch.__version__}")
        log_message(f"CUDA disponibil (is_available): {cuda_available}")
        log_message(f"Număr GPU-uri active: {device_count}")
        log_message(f"Nume GPU selectat: {device_name}")
        log_message(f"Versiune CUDA în PyTorch: {cuda_version}")
    except Exception as e:
        cuda_available = False
        log_message(f"EROARE la importarea PyTorch sau CUDA: {str(e)}")
    
    # If the converted model already exists, launch the app directly
    if os.path.exists(CT2_MODEL_DIR) and (os.path.exists(os.path.join(CT2_MODEL_DIR, "model.bin")) or os.path.exists(os.path.join(CT2_MODEL_DIR, "model.bin.part"))):
        log_message("Modelul local convertit a fost găsit! Se lansează Scriba Ro...")
        import subprocess
        app_script = os.path.join(BASE_DIR, "app.py")
        # Run app.py with pythonw.exe to natively register as a Windows GUI subsystem application
        pythonw_exe = sys.executable.replace("python.exe", "pythonw.exe")
        subprocess.Popen([pythonw_exe, app_script], cwd=BASE_DIR)
        sys.exit(0)
    else:
        log_message("Modelul local convertit nu a fost găsit. Se deschide interfața grafică de configurare (Setup GUI)...")
        # Show setup GUI
        app = SetupGUI(cuda_available)
        app.mainloop()

if __name__ == "__main__":
    main()
