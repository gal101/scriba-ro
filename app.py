import os
import sys

# Guarantee that the application's directory is always in Python's search path,
# regardless of what current working directory the script was launched from!
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Redirect stdout/stderr if running under pythonw.exe to prevent crashes
if sys.stdout is None or sys.stderr is None:
    class NullWriter:
        def write(self, text):
            pass
        def flush(self):
            pass
    if sys.stdout is None:
        sys.stdout = NullWriter()
    if sys.stderr is None:
        sys.stderr = NullWriter()

import json
import threading
import queue
import time
import ctypes

# 1. Register local CUDA DLLs dynamically on Windows before importing faster_whisper
if sys.platform == "win32":
    cuda_dirs = [
        os.path.join(sys.prefix, r"Lib\site-packages\nvidia\cublas\bin"),
        os.path.join(sys.prefix, r"Lib\site-packages\nvidia\cudnn\bin"),
    ]
    for d in cuda_dirs:
        if os.path.isdir(d):
            os.add_dll_directory(d)

import numpy as np
import sounddevice as sd
import pyautogui
import pyperclip
from pynput import keyboard, mouse
from PIL import Image, ImageDraw
import pystray
import tkinter as tk
import customtkinter as ctk

# Import our transparent floating overlay
from overlay import DictationOverlay

# Configuration paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
CT2_MODEL_DIR = os.path.join(BASE_DIR, "models", "whisper-large-v3-turbo-ct2")
LOG_PATH = os.path.join(BASE_DIR, "scriba.log")

def log_message(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [APP] {message}\n"
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

# Global Application State
class AppState:
    def __init__(self):
        self.state = "idle"  # "idle", "recording", "transcribing"
        self.config = {
            "hotkey": "Right Control",
            "mouse_button": "None",
            "mode": "Hold to Talk",
            "add_space": True,
            "input_device": "Default",
            "device": "cpu",
            "quantization": "int8",
            "language": "ro"
        }
        self.model = None       # faster-whisper (engleza)
        self.nemo_model = None  # NeMo ro-fast-conformer (romana)
        self.overlay = None
        self.tray_icon = None
        
        # Audio capturing buffers
        self.audio_queue = queue.Queue()
        self.audio_stream = None
        
        # Threads/Listeners
        self.keyboard_listener = None
        self.mouse_listener = None
        self.hotkey_pressed = False
        
        self.settings_window = None
        self.dashboard = None
        self.mutex = None

app_state = AppState()

# Load/Save Configuration
def load_config():
    import torch
    cuda_available = torch.cuda.is_available()
    
    default_config = {
        "hotkey": "Right Control",
        "mouse_button": "None",
        "mode": "Hold to Talk",
        "add_space": True,
        "input_device": "Default",
        "device": "cuda" if cuda_available else "cpu",
        "quantization": "float16" if cuda_available else "int8",
        "language": "ro"
    }
    
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                loaded = json.load(f)
                # Merge defaults to handle old or corrupted configs
                default_config.update(loaded)
                app_state.config = default_config
        except Exception:
            app_state.config = default_config
    else:
        app_state.config = default_config
        save_config()

def save_config():
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(app_state.config, f, indent=4)
    except Exception as e:
        print(f"Error saving config: {e}")

# Map string settings to pynput Key objects
def get_pynput_key(key_str):
    mapping = {
        "Right Control": keyboard.Key.ctrl_r,
        "Left Control": keyboard.Key.ctrl_l,
        "Right Shift": keyboard.Key.shift_r,
        "Left Shift": keyboard.Key.shift_l,
        "F8": keyboard.Key.f8,
        "F10": keyboard.Key.f10,
        "Scroll Lock": keyboard.Key.scroll_lock,
        "Insert": keyboard.Key.insert,
    }
    return mapping.get(key_str, keyboard.Key.ctrl_r)

def get_pynput_mouse_button(btn_str):
    mapping = {
        "Mouse 4 (Side 1)": mouse.Button.x1,
        "Mouse 5 (Side 2)": mouse.Button.x2,
    }
    return mapping.get(btn_str, None)

def get_device_index(selected_device_str):
    if not selected_device_str or selected_device_str == "Default":
        return None
    try:
        devices = sd.query_devices()
        
        # Priority mapping for host APIs to get the best driver performance
        api_priority = {
            "Windows WASAPI": 3,
            "Windows DirectSound": 2,
            "MME": 1,
            "Windows WDM-KS": 0
        }
        
        best_idx = None
        best_priority = -1
        
        # Look for the device matching the selected clean name
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0 and d['name'] == selected_device_str:
                hostapi_name = sd.query_hostapis(d['hostapi'])['name']
                priority = api_priority.get(hostapi_name, 0)
                if priority > best_priority:
                    best_priority = priority
                    best_idx = i
                    
        if best_idx is not None:
            return best_idx
            
        # Substring matching fallback
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                if d['name'] in selected_device_str:
                    return i
    except Exception as e:
        log_message(f"Eroare la determinarea indexului microfonului: {str(e)}")
    return None

# ─── Model Loaders ────────────────────────────────────────────────────────────

def load_whisper_model():
    """Lazy-load modelul faster-whisper (openai/whisper-large-v3-turbo) pentru engleza."""
    if app_state.model is not None:
        return True
        
    try:
        from faster_whisper import WhisperModel
        log_message(f"Se încarcă modelul Whisper din {CT2_MODEL_DIR} pe dispozitivul {app_state.config['device']} ({app_state.config['quantization']})...")
        app_state.model = WhisperModel(
            CT2_MODEL_DIR, 
            device=app_state.config["device"], 
            compute_type=app_state.config["quantization"]
        )
        log_message("Modelul Whisper (engleza) incarcat cu succes!")
        if app_state.tray_icon:
            try:
                hotkey_info = app_state.config['hotkey']
                if app_state.config['mouse_button'] != "None":
                    hotkey_info += f" / {app_state.config['mouse_button']}"
                app_state.tray_icon.notify(
                    f"Scriba Ro este gata! Apasa {hotkey_info} pentru dictare.",
                    "Scriba Ro - Pregatit"
                )
            except Exception as e_notify:
                log_message(f"Notificarea de pregatire nu a putut fi afisata: {str(e_notify)}")
        return True
    except Exception as e:
        log_message(f"EROARE CRITICA la incarcarea modelului Whisper: {str(e)}")
        if app_state.tray_icon:
            try:
                app_state.tray_icon.notify(
                    f"Eroare la incarcarea modelului: {str(e)}. Verifica setarile.",
                    "Scriba Ro - Eroare"
                )
            except Exception:
                pass
        return False

def find_nemo_file():
    """Gaseste fisierul .nemo al modelului upb-nlp/ro-fast-conformer."""
    # 1. Cauta in cache-ul standard NeMo (de la from_pretrained / restore_from anterior)
    nemo_cache = os.path.join(os.path.expanduser("~"), ".cache", "torch", "NeMo")
    if os.path.isdir(nemo_cache):
        for root, dirs, files in os.walk(nemo_cache):
            for f in files:
                if f.endswith(".nemo") and "ro-fast-conformer" in root:
                    return os.path.join(root, f)
    # 2. Cauta in cache-ul HuggingFace Hub
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if os.path.isdir(hf_cache):
        for root, dirs, files in os.walk(hf_cache):
            for f in files:
                if f.endswith(".nemo") and "ro-fast-conformer" in root:
                    return os.path.join(root, f)
    # 3. Fallback: in folderul local models/
    local = os.path.join(BASE_DIR, "models", "Speech_To_Text_Finetuning.nemo")
    if os.path.exists(local):
        return local
    return None

def load_nemo_model():
    """Lazy-load modelul NeMo (upb-nlp/ro-fast-conformer) pentru romana."""
    if app_state.nemo_model is not None:
        return True

    nemo_path = find_nemo_file()
    if not nemo_path:
        log_message("EROARE: Fisierul .nemo nu a fost gasit. Ruleaza TEST_UPB_MODEL.bat pentru a-l descarca.")
        if app_state.tray_icon:
            try:
                app_state.tray_icon.notify(
                    "Modelul pentru romana nu a fost gasit. Ruleaza TEST_UPB_MODEL.bat.",
                    "Scriba Ro - Model lipsa"
                )
            except Exception:
                pass
        return False

    try:
        import torch
        import nemo.collections.asr as nemo_asr
        log_message(f"Se incarca modelul NeMo din: {os.path.basename(nemo_path)}")
        device = app_state.config.get("device", "cpu")
        app_state.nemo_model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location=torch.device(device))
        app_state.nemo_model.eval()
        log_message(f"Modelul NeMo (ro-fast-conformer) incarcat cu succes pe dispozitivul {device}!")
        if app_state.tray_icon:
            try:
                hotkey_info = app_state.config['hotkey']
                if app_state.config['mouse_button'] != "None":
                    hotkey_info += f" / {app_state.config['mouse_button']}"
                app_state.tray_icon.notify(
                    f"Scriba Ro este gata! Apasa {hotkey_info} pentru dictare in romana.",
                    "Scriba Ro - Pregatit"
                )
            except Exception:
                pass
        return True
    except Exception as e:
        log_message(f"EROARE la incarcarea modelului NeMo: {str(e)}")
        if app_state.tray_icon:
            try:
                app_state.tray_icon.notify(
                    f"Eroare la incarcarea modelului NeMo: {str(e)}",
                    "Scriba Ro - Eroare"
                )
            except Exception:
                pass
        return False

# Audio Recording Callbacks
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status warning: {status}")
    app_state.audio_queue.put(indata.copy())
    
    # Calculate volume level in real-time and pass to overlay visualizer
    try:
        if app_state.overlay and app_state.state == "recording":
            # Calculate Root Mean Square (RMS) of audio chunk
            rms = np.sqrt(np.mean(indata**2))
            # Normal speech RMS is around 0.01 to 0.15. Scale and clip between 0.0 and 1.0.
            vol = min(1.0, rms * 7.5)
            app_state.overlay.update_volume(vol)
    except Exception:
        pass

def start_recording():
    try:
        # Empty the queue
        while not app_state.audio_queue.empty():
            app_state.audio_queue.get()
            
        selected_device = app_state.config.get("input_device", "Default")
        device_idx = get_device_index(selected_device)
        
        # Default sample rate tracker
        app_state.active_samplerate = 16000
        
        log_message(f"Se inițializează microfonul: {selected_device} (index: {device_idx})")
        
        # Try opening at 16000 Hz first
        try:
            app_state.audio_stream = sd.InputStream(
                device=device_idx,
                samplerate=16000,
                channels=1,
                dtype='float32',
                callback=audio_callback
            )
            app_state.audio_stream.start()
            log_message(f"Pornire înregistrare audio de la dispozitiv {selected_device} la 16kHz...")
            return
        except Exception as e_16k:
            log_message(f"Eroare la deschiderea la 16kHz: {str(e_16k)}. Se caută rata nativă...")
            
        # Try querying the device's native default sample rate
        native_sr = 16000
        try:
            if device_idx is not None:
                device_info = sd.query_devices(device_idx, 'input')
                native_sr = int(device_info.get('default_samplerate', 16000))
        except Exception:
            pass
            
        log_message(f"Se încearcă deschiderea microfonului la rata nativă a dispozitivului: {native_sr}Hz")
        try:
            app_state.audio_stream = sd.InputStream(
                device=device_idx,
                samplerate=native_sr,
                channels=1,
                dtype='float32',
                callback=audio_callback
            )
            app_state.audio_stream.start()
            app_state.active_samplerate = native_sr
            log_message(f"Pornire înregistrare audio la rata nativă {native_sr}Hz (resamplare activă)...")
            return
        except Exception as e_native:
            log_message(f"Eroare la deschiderea la rata nativă: {str(e_native)}")
            
        # Fallback to system default input device at 16000 Hz
        log_message("FALLBACK: Se încearcă deschiderea microfonului implicit al sistemului la 16kHz")
        try:
            app_state.audio_stream = sd.InputStream(
                device=None,
                samplerate=16000,
                channels=1,
                dtype='float32',
                callback=audio_callback
            )
            app_state.audio_stream.start()
            app_state.active_samplerate = 16000
            log_message("Pornire înregistrare de la microfonul implicit la 16kHz...")
            return
        except Exception as e_def_16k:
            log_message(f"Eroare la fallback implicit 16kHz: {str(e_def_16k)}")
            
        # Final fallback: system default device at its native sample rate
        default_device_info = sd.query_devices(None, 'input')
        default_sr = int(default_device_info.get('default_samplerate', 44100))
        log_message(f"FALLBACK FINAL: Se încearcă deschiderea microfonului implicit la rata nativă {default_sr}Hz")
        app_state.audio_stream = sd.InputStream(
            device=None,
            samplerate=default_sr,
            channels=1,
            dtype='float32',
            callback=audio_callback
        )
        app_state.audio_stream.start()
        app_state.active_samplerate = default_sr
        log_message(f"Pornire înregistrare de la microfonul implicit la rata nativă {default_sr}Hz...")
        
    except Exception as e:
        log_message(f"EROARE CRITICĂ la pornirea fluxului audio: {str(e)}")

def stop_recording_and_get_audio():
    if app_state.audio_stream is not None:
        try:
            app_state.audio_stream.stop()
            app_state.audio_stream.close()
            app_state.audio_stream = None
            log_message("Oprire înregistrare audio.")
        except Exception as e:
            log_message(f"EROARE la oprirea fluxului audio: {str(e)}")
            
    chunks = []
    while not app_state.audio_queue.empty():
        chunks.append(app_state.audio_queue.get())
        
    if not chunks:
        log_message("AVERTISMENT: Nu s-a capturat niciun fragment audio!")
        return np.array([], dtype=np.float32)
        
    audio_data = np.concatenate(chunks, axis=0).flatten()
    
    # Resample to 16000 Hz if recorded at a native rate other than 16000 Hz
    active_sr = getattr(app_state, 'active_samplerate', 16000)
    if active_sr != 16000:
        log_message(f"Resamplare audio de la {active_sr}Hz la 16000Hz...")
        try:
            import scipy.signal
            num_samples = int(len(audio_data) * 16000 / active_sr)
            audio_data = scipy.signal.resample(audio_data, num_samples)
            log_message(f"Resamplare finalizată cu succes. Dimensiune finală: {len(audio_data)} eșantioane.")
        except Exception as e_resample:
            log_message(f"Eroare la resamplare scipy: {str(e_resample)}")
            # Fallback simple linear interpolation if scipy fails
            try:
                xp = np.linspace(0, len(audio_data), len(audio_data))
                new_len = int(len(audio_data) * 16000 / active_sr)
                x = np.linspace(0, len(audio_data), new_len)
                audio_data = np.interp(x, xp, audio_data)
                log_message("Resamplare liniară (numpy.interp) finalizată.")
            except Exception as e_interp:
                log_message(f"Eroare la resamplare liniară: {str(e_interp)}")
                
    log_message(f"Fragment audio capturat în memorie: {len(audio_data)} eșantioane (~{len(audio_data)/16000:.2f} secunde).")
    return audio_data

# Speech processing & typing
def transcribe_and_type_thread(audio_data):
    try:
        lang = app_state.config.get("language", "ro")
        log_message(f"Etapa: Transcriere locala in curs... (limba: {lang})")
        start_time = time.time()
        text = ""

        if lang == "ro":
            # ── Romana: folosim NeMo ro-fast-conformer ────────────────────────
            if app_state.nemo_model is None:
                success = load_nemo_model()
                if not success:
                    return

            # NeMo necesita fisier WAV pe disc, nu array numpy
            import wave
            tmp_wav = os.path.join(BASE_DIR, "_tmp_dictation.wav")
            audio_int16 = (audio_data * 32767).astype('int16')
            with wave.open(tmp_wav, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)   # 16-bit
                wf.setframerate(16000)
                wf.writeframes(audio_int16.tobytes())

            result = app_state.nemo_model.transcribe([tmp_wav])

            # Curata fisierul temporar
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

            # NeMo poate returna string sau obiect cu .text
            if isinstance(result, list) and result:
                text = result[0]
                if hasattr(text, 'text'):
                    text = text.text
                text = str(text).strip()
            else:
                text = str(result).strip()

            duration = time.time() - start_time
            log_message(f"NeMo transcriere completa in {duration:.2f}s: '{text}'")

        else:
            # ── Engleza (sau alta limba): folosim Whisper ─────────────────────
            if app_state.model is None:
                success = load_whisper_model()
                if not success:
                    return

            lang_code = None if lang == "auto" else lang
            segments, info = app_state.model.transcribe(
                audio_data, language=lang_code, beam_size=5, vad_filter=True
            )
            text = "".join([segment.text for segment in segments]).strip()
            duration = time.time() - start_time
            log_message(f"Whisper transcriere completa in {duration:.2f}s: '{text}' (detectat: {info.language}, prob: {info.language_probability:.2f})")

        if text:
            inject_text(text, add_space=app_state.config["add_space"])
        else:
            log_message("Niciun cuvant detectat in audio.")

    except Exception as e:
        log_message(f"EROARE in timpul transcrierii vocale: {str(e)}")

    finally:
        app_state.overlay.hide()
        app_state.state = "idle"
        if app_state.dashboard:
            app_state.dashboard.update_status("idle")

def inject_text(text, add_space=True):
    if not text:
        return
        
    # Backup current clipboard
    old_clipboard = pyperclip.paste()
    log_message("Salvare backup clipboard curent.")
    
    # Prepend space if selected
    if add_space:
        text = " " + text
        
    # Copy to clipboard
    pyperclip.copy(text)
    
    # Simulate Ctrl+V to paste instantly with correct diacritics
    log_message(f"Trimitere text prin simulare taste (Ctrl+V): '{text}'")
    pyautogui.hotkey('ctrl', 'v')
    
    # Restore clipboard in a tiny background delay so we don't hold the lock
    def restore():
        time.sleep(0.3)
        pyperclip.copy(old_clipboard)
        log_message("Restaurare clipboard original finalizată.")
        
    threading.Thread(target=restore, daemon=True).start()

# Global Input Event Handling
def toggle_language_callback():
    """Toggles dictation language, updates all UI elements, and launches preloading."""
    current_lang = app_state.config.get("language", "ro")
    new_lang = "en" if current_lang == "ro" else "ro"
    app_state.config["language"] = new_lang
    log_message(f"Schimbare limbă prin comutator/F8: {current_lang} -> {new_lang}")
    
    # Save the new language config
    save_config()
    
    # Preload the newly selected language's model in background (non-blocking)
    check_and_preload_active_model()
    
    # Synchronize Settings window dropdown if open
    if app_state.settings_window and app_state.settings_window.winfo_exists():
        try:
            lang_map_reverse = {
                "ro": "\U0001f1f7\U0001f1f4  Român\u0103 \u2014 NeMo UPB",
                "en": "\U0001f1ec\U0001f1e7  Englez\u0103 \u2014 Whisper"
            }
            app_state.settings_window.lang_menu.set(lang_map_reverse.get(new_lang, "\U0001f1f7\U0001f1f4  Român\u0103 \u2014 NeMo UPB"))
        except Exception:
            pass
            
    # Update real-time Dashboard status
    if app_state.dashboard:
        is_model_loaded = (app_state.nemo_model is not None) if new_lang == "ro" else (app_state.model is not None)
        if not is_model_loaded:
            app_state.dashboard.update_status("loading_model")
        else:
            app_state.dashboard.update_status("idle")
        app_state.dashboard.update_lang_button(new_lang)
            
    # Update overlay UI switch appearance
    if app_state.overlay:
        app_state.overlay.update_language_ui(new_lang)
        
    # Show active language notification in tray
    if app_state.tray_icon:
        try:
            title = "Scriba Ro - Limbă Schimbată"
            msg = "Limbă activă: Română (NeMo UPB)" if new_lang == "ro" else "Active Language: English (Whisper)"
            app_state.tray_icon.notify(msg, title)
        except Exception:
            pass

def on_trigger_down():
    if app_state.state == "idle":
        app_state.state = "recording"
        app_state.overlay.show_listening(app_state.config.get("language", "ro"))
        if app_state.dashboard:
            app_state.dashboard.update_status("recording")
        start_recording()
    elif app_state.state == "recording" and app_state.config["mode"] == "Press to Toggle":
        # Toggle mode: press again to stop
        on_trigger_up()

def on_trigger_up():
    if app_state.state == "recording":
        app_state.state = "transcribing"
        app_state.overlay.show_transcribing(app_state.config.get("language", "ro"))
        if app_state.dashboard:
            app_state.dashboard.update_status("transcribing")
        
        # Stop mic and get audio array in memory
        audio_data = stop_recording_and_get_audio()
        
        # If recording was too short, ignore
        if len(audio_data) < 8000: # Less than 0.5 seconds
            log_message("Înregistrare prea scurtă. Se ignoră.")
            app_state.overlay.hide()
            app_state.state = "idle"
            if app_state.dashboard:
                app_state.dashboard.update_status("idle")
            return
            
        # Transcribe and type in background thread to avoid locking GUI
        threading.Thread(target=transcribe_and_type_thread, args=(audio_data,), daemon=True).start()

# Pynput Listeners
def on_key_press(key):
    # F8 global keyboard shortcut to toggle dictation language at any time!
    if key == keyboard.Key.f8:
        toggle_language_callback()
        return

    target_key = get_pynput_key(app_state.config["hotkey"])
    if key == target_key:
        if not app_state.hotkey_pressed:
            app_state.hotkey_pressed = True
            on_trigger_down()

def on_key_release(key):
    target_key = get_pynput_key(app_state.config["hotkey"])
    if key == target_key:
        app_state.hotkey_pressed = False
        if app_state.config["mode"] == "Hold to Talk":
            on_trigger_up()

def on_mouse_click(x, y, button, pressed):
    target_button = get_pynput_mouse_button(app_state.config["mouse_button"])
    if target_button and button == target_button:
        if pressed:
            on_trigger_down()
        else:
            if app_state.config["mode"] == "Hold to Talk":
                on_trigger_up()

def restart_listeners():
    # Stop existing listeners
    if app_state.keyboard_listener:
        app_state.keyboard_listener.stop()
    if app_state.mouse_listener:
        app_state.mouse_listener.stop()
        
    # Start new keyboard listener
    app_state.keyboard_listener = keyboard.Listener(
        on_press=on_key_press,
        on_release=on_key_release
    )
    app_state.keyboard_listener.daemon = True  # Make daemon so it exits with main thread
    app_state.keyboard_listener.start()
    
    # Start mouse listener if side buttons selected
    if app_state.config["mouse_button"] != "None":
        app_state.mouse_listener = mouse.Listener(on_click=on_mouse_click)
        app_state.mouse_listener.daemon = True  # Make daemon so it exits with main thread
        app_state.mouse_listener.start()

# Settings GUI (CustomTkinter Toplevel)
class SettingsWindow(ctk.CTkToplevel):
    def __init__(self):
        super().__init__()
        
        self.title("Scriba Ro - Setări")
        self.geometry("560x540")
        self.resizable(False, False)
        
        # Ensure it stays on top of parent window
        self.attributes("-topmost", True)
        ctk.set_appearance_mode("dark")
        
        # Title
        self.title_label = ctk.CTkLabel(
            self, 
            text="Setări Scriba Ro", 
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold")
        )
        self.title_label.pack(pady=15)
        
        # Grid frame
        self.grid_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.grid_frame.pack(fill="both", expand=True, padx=45)
        self.grid_frame.grid_columnconfigure(0, weight=1)
        self.grid_frame.grid_columnconfigure(1, weight=1)
        
        # 1. Hotkey Selector
        self.hk_label = ctk.CTkLabel(
            self.grid_frame, 
            text="Tastă Rapidă (Hotkey):", 
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            anchor="w"
        )
        self.hk_label.grid(row=0, column=0, pady=12, padx=10, sticky="w")
        
        self.hk_menu = ctk.CTkOptionMenu(
            self.grid_frame,
            values=["Right Control", "Left Control", "F10", "Scroll Lock", "Insert"],
            command=self.on_setting_changed
        )
        self.hk_menu.set(app_state.config["hotkey"])
        self.hk_menu.grid(row=0, column=1, pady=12, padx=10, sticky="e")
        
        # 2. Mouse Button Selector
        self.mouse_label = ctk.CTkLabel(
            self.grid_frame, 
            text="Buton Mouse (Opțional):", 
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            anchor="w"
        )
        self.mouse_label.grid(row=1, column=0, pady=12, padx=10, sticky="w")
        
        self.mouse_menu = ctk.CTkOptionMenu(
            self.grid_frame,
            values=["None", "Mouse 4 (Side 1)", "Mouse 5 (Side 2)"],
            command=self.on_setting_changed
        )
        self.mouse_menu.set(app_state.config["mouse_button"])
        self.mouse_menu.grid(row=1, column=1, pady=12, padx=10, sticky="e")
        
        # 3. Activation Mode
        self.mode_label = ctk.CTkLabel(
            self.grid_frame, 
            text="Mod Activare:", 
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            anchor="w"
        )
        self.mode_label.grid(row=2, column=0, pady=12, padx=10, sticky="w")
        
        self.mode_menu = ctk.CTkOptionMenu(
            self.grid_frame,
            values=["Hold to Talk", "Press to Toggle"],
            command=self.on_setting_changed
        )
        self.mode_menu.set(app_state.config["mode"])
        self.mode_menu.grid(row=2, column=1, pady=12, padx=10, sticky="e")
        
        # 4. Spacing Checkbox
        self.space_label = ctk.CTkLabel(
            self.grid_frame, 
            text="Spațiere Inteligentă:", 
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            anchor="w"
        )
        self.space_label.grid(row=3, column=0, pady=12, padx=10, sticky="w")
        
        self.space_var = ctk.BooleanVar(value=app_state.config["add_space"])
        self.space_chk = ctk.CTkCheckBox(
            self.grid_frame, 
            text="Adaugă spațiu înaintea textului",
            variable=self.space_var,
            command=self.on_setting_changed
        )
        self.space_chk.grid(row=3, column=1, pady=12, padx=10, sticky="w")
        
        # 5. Microphone Selector
        self.mic_dev_label = ctk.CTkLabel(
            self.grid_frame, 
            text="Microfon:", 
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            anchor="w"
        )
        self.mic_dev_label.grid(row=4, column=0, pady=12, padx=10, sticky="w")
        
        # Populate input devices with simple, clean unique names (exactly like Discord!)
        self.dev_list = ["Default"]
        wasapi_names = []
        mme_names = []
        try:
            devices = sd.query_devices()
            for d in devices:
                if d['max_input_channels'] > 0:
                    hostapi_name = sd.query_hostapis(d['hostapi'])['name']
                    name = d['name']
                    
                    # Filter out obvious system/generic loopbacks and output devices
                    name_lower = name.lower()
                    if any(x in name_lower for x in ["mapper", "primary", "stereo mix", "reverb", "loopback", "aux"]):
                        continue
                        
                    # Filter out raw registry paths or garbage strings (e.g. from WDM-KS)
                    if any(x in name for x in ["@", "%", "#", ";", "\\", "/"]):
                        continue
                        
                    # Filter out generic device labels
                    if name in ["Input", "Output", "Primary Sound Capture Driver", "Microsoft Sound Mapper - Input"]:
                        continue
                        
                    # Skip common virtual outputs that aren't microphones
                    if any(x in name_lower for x in ["speaker", "difuzor", "headphones"]):
                        if "microphone" not in name_lower and "mic" not in name_lower:
                            continue
                    
                    if hostapi_name == "Windows WASAPI":
                        if name not in wasapi_names:
                            wasapi_names.append(name)
                    elif hostapi_name == "MME":
                        if name not in mme_names:
                            mme_names.append(name)
            
            # Add WASAPI devices first (beautiful full names)
            for name in wasapi_names:
                self.dev_list.append(name)
                
            # Add MME devices only if they are not already represented by a WASAPI device
            for name in mme_names:
                is_duplicate = False
                for w_name in wasapi_names:
                    # If one name is a prefix/truncation of the other
                    if name.startswith(w_name) or w_name.startswith(name):
                        is_duplicate = True
                        break
                if not is_duplicate and name not in self.dev_list:
                    self.dev_list.append(name)
                    
        except Exception as e:
            log_message(f"Eroare la listarea microfoanelor: {e}")
            
        self.mic_dev_menu = ctk.CTkOptionMenu(
            self.grid_frame,
            values=self.dev_list,
            command=self.on_setting_changed,
            width=220
        )
        selected_device = app_state.config.get("input_device", "Default")
        if selected_device not in self.dev_list:
            selected_device = "Default"
        self.mic_dev_menu.set(selected_device)
        self.mic_dev_menu.grid(row=4, column=1, pady=12, padx=10, sticky="e")
        
        # 6. Language Selector
        self.lang_label = ctk.CTkLabel(
            self.grid_frame, 
            text="Limbă dictare:", 
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            anchor="w"
        )
        self.lang_label.grid(row=5, column=0, pady=12, padx=10, sticky="w")
        
        self.lang_menu = ctk.CTkOptionMenu(
            self.grid_frame,
            values=["\U0001f1f7\U0001f1f4  Român\u0103 \u2014 NeMo UPB", "\U0001f1ec\U0001f1e7  Englez\u0103 \u2014 Whisper"],
            command=self.on_setting_changed,
            width=220
        )
        lang_map_reverse = {"ro": "\U0001f1f7\U0001f1f4  Român\u0103 \u2014 NeMo UPB", "en": "\U0001f1ec\U0001f1e7  Englez\u0103 \u2014 Whisper"}
        self.lang_menu.set(lang_map_reverse.get(app_state.config.get("language", "ro"), "\U0001f1f7\U0001f1f4  Român\u0103 \u2014 NeMo UPB"))
        self.lang_menu.grid(row=5, column=1, pady=12, padx=10, sticky="e")
        
        # Hardware optimization stats
        self.hw_frame = ctk.CTkFrame(self, fg_color="#1E1E24", border_width=1, border_color="#33333C")
        self.hw_frame.pack(fill="x", padx=45, pady=15)
        
        import torch
        dev = "CUDA (Placă Video NVIDIA)" if app_state.config["device"] == "cuda" else "CPU (Procesor)"
        self.hw_label = ctk.CTkLabel(
            self.hw_frame,
            text=f"Hardware Activ: {dev} | Precizie: {app_state.config['quantization']}",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#4CAF50"
        )
        self.hw_label.pack(pady=10)
        
        # Action Buttons
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(fill="x", padx=45, pady=(10, 20))
        
        self.save_btn = ctk.CTkButton(
            self.btn_frame, 
            text="Salvează", 
            width=200, 
            height=38,
            command=self.save_and_close
        )
        self.save_btn.pack(side=tk.LEFT, padx=10, expand=True)
        
        self.close_btn = ctk.CTkButton(
            self.btn_frame, 
            text="Închide", 
            fg_color="#33333C",
            hover_color="#44444F",
            width=200, 
            height=38,
            command=self.destroy
        )
        self.close_btn.pack(side=tk.RIGHT, padx=10, expand=True)
        
    def on_setting_changed(self, *args):
        # Dynamically read and update temporary app config
        app_state.config["hotkey"] = self.hk_menu.get()
        app_state.config["mouse_button"] = self.mouse_menu.get()
        app_state.config["mode"] = self.mode_menu.get()
        app_state.config["add_space"] = self.space_var.get()
        app_state.config["input_device"] = self.mic_dev_menu.get()
        lang_map = {
            "\U0001f1f7\U0001f1f4  Român\u0103 \u2014 NeMo UPB": "ro",
            "\U0001f1ec\U0001f1e7  Englez\u0103 \u2014 Whisper": "en"
        }
        app_state.config["language"] = lang_map.get(self.lang_menu.get(), "ro")
        
    def save_and_close(self):
        save_config()
        restart_listeners()
        self.destroy()
        
        # Preload the newly selected model in background (non-blocking)
        check_and_preload_active_model()

def open_settings_main_thread():
    try:
        if app_state.settings_window is None or not app_state.settings_window.winfo_exists():
            app_state.settings_window = SettingsWindow()
            app_state.settings_window.deiconify()
            app_state.settings_window.lift()
            app_state.settings_window.focus()
        else:
            app_state.settings_window.deiconify()
            app_state.settings_window.lift()
            app_state.settings_window.focus()
    except Exception as e:
        import traceback
        log_message(f"EROARE la deschiderea ferestrei Setări: {str(e)}\n{traceback.format_exc()}")

def open_settings():
    if app_state.dashboard:
        try:
            app_state.dashboard.after(0, open_settings_main_thread)
        except Exception:
            open_settings_main_thread()
    else:
        open_settings_main_thread()

def show_about():
    # Simple win32 popup dialog to avoid complex Tkinter windows for simple information
    ctypes.windll.user32.MessageBoxW(
        0, 
        "Scriba Ro v1.2.0\n\nAsistent local și gratuit de dictare vocală offline.\n\nFolosește modelul: openai/whisper-large-v3-turbo\nLimbi: Română, Engleză, Auto-detect (99 limbi)\nCreat cu drag de Antigravity AI.", 
        "Despre Scriba Ro", 
        0x40 | 0x0
    )

def exit_app():
    log_message("Ieșire din aplicația Scriba Ro...")
    if app_state.keyboard_listener:
        app_state.keyboard_listener.stop()
    if app_state.mouse_listener:
        app_state.mouse_listener.stop()
    if app_state.overlay:
        app_state.overlay.close()
    if app_state.dashboard:
        try:
            app_state.dashboard.destroy()
        except Exception:
            pass
    if app_state.tray_icon:
        app_state.tray_icon.stop()
    sys.exit(0)

class DashboardWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Scriba Ro")
        self.geometry("320x240")
        self.resizable(False, False)
        ctk.set_appearance_mode("dark")
        
        # Position at bottom-right corner of screen (above taskbar)
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = screen_width - 340
        y = screen_height - 320
        self.geometry(f"320x240+{x}+{y}")
        
        # Minimize to tray instead of closing on X button
        self.protocol("WM_DELETE_WINDOW", self.hide_window)
        
        # Header
        self.header_label = ctk.CTkLabel(
            self, 
            text="🎤 Scriba Ro Asistent", 
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold")
        )
        self.header_label.pack(pady=(10, 3))
        
        # Status card
        self.status_card = ctk.CTkFrame(self, fg_color="#1E1E24", border_width=1, border_color="#33333C", height=55)
        self.status_card.pack(fill="x", padx=20, pady=(0, 4))
        self.status_card.pack_propagate(False)
        
        self.status_label = ctk.CTkLabel(
            self.status_card,
            text="🟡 Se încarcă modelul...",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#FF9800"
        )
        self.status_label.pack(pady=(8, 1))
        
        self.info_label = ctk.CTkLabel(
            self.status_card,
            text="Așteptare preîncărcare model lingvistic local...",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#888888"
        )
        self.info_label.pack()

        # ── Language toggle button ───────────────────────────────────────────────
        initial_lang = app_state.config.get("language", "ro")
        lang_text, lang_color = self._lang_btn_style(initial_lang)
        self.lang_btn = ctk.CTkButton(
            self,
            text=lang_text,
            fg_color=lang_color,
            hover_color=self._lang_btn_hover(initial_lang),
            height=32,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            command=self._on_lang_toggle
        )
        self.lang_btn.pack(fill="x", padx=20, pady=(0, 6))
        
        # Buttons row
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(fill="x", padx=20, pady=(0, 10))

        
        self.settings_btn = ctk.CTkButton(
            self.btn_frame,
            text="Setări",
            width=80,
            height=28,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=open_settings
        )
        self.settings_btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        self.hide_btn = ctk.CTkButton(
            self.btn_frame,
            text="Ascunde",
            fg_color="#33333C",
            hover_color="#44444F",
            width=80,
            height=28,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=self.hide_window
        )
        self.hide_btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        self.exit_btn = ctk.CTkButton(
            self.btn_frame,
            text="Ieșire",
            fg_color="#A52A2A",
            hover_color="#8B0000",
            width=80,
            height=28,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=exit_app
        )
        self.exit_btn.pack(side=tk.LEFT, padx=5, expand=True)
        
    def _lang_btn_style(self, lang):
        if lang == "ro":
            return "🇷🇴 Română (NeMo UPB)  —  apasă pentru EN", "#2E7D32"
        else:
            return "🇬🇧 English (Whisper)  —  press for RO", "#1565C0"

    def _lang_btn_hover(self, lang):
        return "#388E3C" if lang == "ro" else "#1976D2"

    def _on_lang_toggle(self):
        toggle_language_callback()

    def update_lang_button(self, lang):
        """Update the language toggle button to reflect the new active language."""
        def _do():
            text, color = self._lang_btn_style(lang)
            hover = self._lang_btn_hover(lang)
            self.lang_btn.configure(text=text, fg_color=color, hover_color=hover)
        try:
            self.after(0, _do)
        except Exception:
            pass

    def hide_window(self):
        log_message("Minimizare panou control în system tray.")
        self.withdraw()
        if app_state.tray_icon:
            app_state.tray_icon.notify("Scriba Ro rulează ascuns în system tray. Click stânga pe pictogramă pentru afișare.", "Scriba Ro - Minimizat")
            
    def show_window(self):
        log_message("Afișare panou control.")
        self.deiconify()
        self.lift()
        self.focus()
        
    def update_status(self, state, text_info=None):
        def _update():
            if state == "idle":
                self.status_label.configure(text="🟢 Scriba Ro este pregătit", text_color="#4CAF50")
                hotkey_info = app_state.config['hotkey']
                if app_state.config['mouse_button'] != "None":
                    hotkey_info += f" / {app_state.config['mouse_button']}"
                self.info_label.configure(text=f"Apasă {hotkey_info} pentru dictare")
            elif state == "loading_model":
                self.status_label.configure(text="🟡 Se încarcă modelul...", text_color="#FF9800")
                self.info_label.configure(text="Pregătire model lingvistic local...")
            elif state == "recording":
                self.status_label.configure(text="🔴 Înregistrez audio...", text_color="#FF5252")
                lang = app_state.config.get("language", "ro")
                prompt_text = "Vorbește acum în română..." if lang == "ro" else "Speak now in English..."
                self.info_label.configure(text=prompt_text)
            elif state == "transcribing":
                self.status_label.configure(text="⏳ Se transcrie audio...", text_color="#FF9800")
                self.info_label.configure(text="Procesare rapidă locală...")
                
        try:
            self.after(0, _update)
        except Exception:
            pass

def show_dashboard():
    if app_state.dashboard:
        app_state.dashboard.after(0, app_state.dashboard.show_window)

def check_and_preload_active_model():
    """Verifică limba activă și încarcă modelul potrivit în fundal dacă nu este încărcat."""
    lang = app_state.config.get("language", "ro")
    if lang == "ro":
        if app_state.nemo_model is None:
            if app_state.dashboard:
                app_state.dashboard.update_status("loading_model")
                
            def load_job():
                success = load_nemo_model()
                if success and app_state.dashboard:
                    app_state.dashboard.update_status("idle")
            
            threading.Thread(target=load_job, daemon=True).start()
    else:
        if app_state.model is None:
            if app_state.dashboard:
                app_state.dashboard.update_status("loading_model")
                
            def load_job():
                success = load_whisper_model()
                if success and app_state.dashboard:
                    app_state.dashboard.update_status("idle")
            
            threading.Thread(target=load_job, daemon=True).start()

def preload_model_and_listeners(icon):
    log_message("Inițializare ascultători taste/mouse...")
    # Pre-start listeners
    restart_listeners()
    log_message("Ascultători (listeners) taste/mouse porniți.")
    
    # Load model of active language in background thread
    def show_notification():
        time.sleep(1.0) # Wait for pystray to finish registering the tray icon
        try:
            lang = app_state.config.get("language", "ro")
            msg = "Scriba Ro pornește... Se încarcă modelul lingvistic pentru română." if lang == "ro" else "Scriba Ro pornește... Se încarcă modelul lingvistic pentru engleză."
            icon.notify(msg, "Scriba Ro - Pornire")
        except Exception as e_notify:
            log_message(f"Notificarea de pornire nu a putut fi afișată: {str(e_notify)}")
            
    threading.Thread(target=show_notification, daemon=True).start()
        
    log_message("Inițiere încărcare model activ în fundal...")
    check_and_preload_active_model()

def check_single_instance():
    # A unique name for our Windows mutex
    mutex_name = "Global\\ScribaRo_SingleInstance_Mutex"
    try:
        kernel32 = ctypes.windll.kernel32
        CreateMutexW = kernel32.CreateMutexW
        CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
        CreateMutexW.restype = ctypes.c_void_p
        
        GetLastError = kernel32.GetLastError
        GetLastError.argtypes = []
        GetLastError.restype = ctypes.c_ulong
        
        app_state.mutex = CreateMutexW(None, True, mutex_name)
        if GetLastError() == 183: # ERROR_ALREADY_EXISTS
            return False
    except Exception:
        pass
    return True

def hide_console():
    # Programmatically hide console window on Windows to run silently while remaining interactive (prevents tray icon hiding bug)
    try:
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0) # SW_HIDE = 0
    except Exception:
        pass

def main():
    # 1. Hide console window programmatically to run silently
    if sys.platform == "win32":
        hide_console()

    # 2. Single Instance Protection (stops double launches and double typing)
    if sys.platform == "win32" and not check_single_instance():
        try:
            ctypes.windll.user32.MessageBoxW(
                0, 
                "Scriba Ro rulează deja în fundal!\nVerifică pictograma roșie din dreapta-jos (lângă ceas).", 
                "Scriba Ro - Deja activ", 
                0x30 | 0x0
            )
        except Exception:
            pass
        sys.exit(0)

    # Only clear the log file if it hasn't been written to in the last 10 seconds
    # (which prevents erasing logs from setup.py when it launches app.py directly)
    log_file_exists = os.path.exists(LOG_PATH)
    is_recent = False
    if log_file_exists:
        try:
            mtime = os.path.getmtime(LOG_PATH)
            if time.time() - mtime < 10:
                is_recent = True
        except Exception:
            pass
            
    if not is_recent:
        try:
            with open(LOG_PATH, "w", encoding="utf-8") as f:
                f.write("")
        except Exception:
            pass

    log_message("==========================================")
    log_message("     PORNIRE APLICAȚIE PRINCIPALĂ SCRIBA RO")
    log_message("==========================================")
    log_message(f"Executabil Python: {sys.executable}")
    
    # Load user config
    load_config()
    log_message(f"Configurație încărcată: {app_state.config}")
    
    # NOTE: NeMo pre-import is done inside preload_model_and_listeners (tray thread)
    # to avoid blocking the main thread before Tkinter/dashboard is created.
    
    # Initialize tray icon natively in a dedicated thread with COM initialization and robust error logging
    try:
        # Transparent background matching Windows Shell specs (32x32 standard size for High-DPI screens)
        image = Image.new('RGBA', (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        # Beautiful circular red button (padded by 4 pixels)
        draw.ellipse([4, 4, 28, 28], fill=(229, 57, 53, 255))
        # Draw simple white microphone shape inside (centered)
        draw.rounded_rectangle([12, 9, 20, 19], radius=3, fill=(255, 255, 255, 255))
        draw.line([16, 19, 16, 23], fill=(255, 255, 255, 255), width=2)
        draw.line([11, 23, 21, 23], fill=(255, 255, 255, 255), width=2)
        
        menu = pystray.Menu(
            pystray.MenuItem("Panou Control (Dashboard)", show_dashboard),
            pystray.MenuItem("Setări (Settings)", open_settings),
            pystray.MenuItem("Despre (About)", show_about),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Ieșire (Exit)", exit_app)
        )
        
        app_state.tray_icon = pystray.Icon(
            "Scriba Ro", 
            image, 
            "Scriba Ro - Dictare offline în Română", 
            menu
        )
        
        def run_tray():
            # Initialize COM/OLE on this dedicated thread to guarantee perfect Shell tray icon registration!
            try:
                import ctypes
                ctypes.windll.ole32.OleInitialize(None)
                log_message("COM/OLE inițializat cu succes pe thread-ul dedicat System Tray.")
            except Exception as e_ole:
                log_message(f"AVERTISMENT la inițializarea COM/OLE: {str(e_ole)}")
                
            try:
                log_message("Se pornește bucla System Tray în thread dedicat...")
                app_state.tray_icon.run(setup=preload_model_and_listeners)
            except Exception as e_tray_loop:
                log_message(f"EROARE CRITICĂ în thread-ul System Tray: {str(e_tray_loop)}")
                import traceback
                log_message(traceback.format_exc())
                
        log_message("Lansare thread dedicat pentru pictograma System Tray...")
        tray_thread = threading.Thread(target=run_tray, daemon=True)
        tray_thread.start()
    except Exception as e_tray_init:
        log_message(f"EROARE CRITICĂ la pregătirea System Tray: {str(e_tray_init)}")
    
    # Start dashboard CTk loop on main thread (displays as native Windows application)
    app_state.dashboard = DashboardWindow()
    
    # Initialize overlay directly on the main thread context, passing dashboard as parent!
    app_state.overlay = DictationOverlay(
        parent=app_state.dashboard, 
        on_language_toggle=toggle_language_callback,
        initial_lang=app_state.config.get("language", "ro")
    )
    
    app_state.dashboard.mainloop()

if __name__ == "__main__":
    main()
