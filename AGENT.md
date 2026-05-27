# AGENT.md — Developer & AI Agent Reference for Scriba Ro

Read this document in full before making any changes to the codebase.
It describes the architecture, data flow, files, and key technical decisions of **Scriba Ro**.

---

## 1. What the app does

Scriba Ro is a **Windows offline voice dictation app** that lives in the system tray.
The user holds a hotkey, speaks, and the transcribed text is automatically injected
into the active window via clipboard (simulated Ctrl+V).

There are **two ASR engines**, selected based on the active language:

| Language | Model | Library | File |
|----------|-------|---------|------|
| 🇷🇴 Romanian | `upb-nlp/ro-fast-conformer` | NeMo 2.7.3 | `models/Speech_To_Text_Finetuning.nemo` |
| 🇬🇧 English | `openai/whisper-large-v3-turbo` | faster-whisper | `models/whisper-large-v3-turbo-ct2/` |

---

## 2. File structure

```
Whisper/
├── app.py                  # Main logic (entry point)
├── overlay.py              # Floating indicator window (Tkinter)
├── setup.py                # First-run setup wizard
├── run.bat                 # Launcher: runs setup.py, then app.py
├── requirements.txt        # pip dependencies
├── config.json             # Persisted user config (gitignored)
├── scriba.log              # Runtime log (gitignored)
├── models/
│   ├── whisper-large-v3-turbo-ct2/    # Whisper model (CTranslate2 format)
│   └── Speech_To_Text_Finetuning.nemo # NeMo UPB model (Romanian)
└── .venv/                  # Python virtual environment (gitignored)
```

---

## 3. Main files

### `app.py` — core logic

Contains all application components.

#### `AppState` (global singleton)
```python
app_state = AppState()
```
Stores all global state:
- `app_state.model` — faster-whisper instance (Whisper, for EN)
- `app_state.nemo_model` — NeMo ASRModel instance (for RO)
- `app_state.config` — dict with current user configuration
- `app_state.state` — `"idle"` / `"recording"` / `"transcribing"`
- `app_state.overlay` — `DictationOverlay` instance
- `app_state.dashboard` — `DashboardWindow` instance
- `app_state.tray_icon` — pystray instance
- `app_state.keyboard_listener`, `app_state.mouse_listener`

#### Key functions

| Function | Purpose |
|----------|---------|
| `load_config()` / `save_config()` | Read/write `config.json` |
| `find_nemo_file()` | Searches for `.nemo` file in `models/` first, then `~/.cache/torch/NeMo/` |
| `load_nemo_model()` | Loads NeMo model onto CUDA/CPU, sets `app_state.nemo_model` |
| `load_whisper_model()` | Loads Whisper model, sets `app_state.model` |
| `check_and_preload_active_model()` | Detects active language and launches the correct model loader on a daemon thread |
| `transcribe_and_type_thread(audio_data)` | Routes transcription to NeMo (RO) or Whisper (EN), injects text |
| `start_recording()` | Starts `sd.InputStream` at 16kHz |
| `stop_recording_and_get_audio()` | Collects chunks from `audio_queue`, resamples if needed |
| `inject_text(text, add_space)` | Copies to clipboard and simulates Ctrl+V |
| `toggle_language_callback()` | Switches language RO↔EN, saves config, preloads new model, updates all UI elements |
| `restart_listeners()` | Restarts keyboard/mouse listeners after hotkey change |
| `preload_model_and_listeners(icon)` | Runs on tray thread after startup: initializes listeners and triggers model preload |
| `main()` | Entry point: single-instance check, tray icon creation, dashboard startup, mainloop |

#### `SettingsWindow` (CTkToplevel)
- Language dropdown maps `"🇷🇴 Romanian — NeMo UPB"` ↔ `"ro"` and `"🇬🇧 English — Whisper"` ↔ `"en"`
- `save_and_close()` saves config, restarts listeners, destroys the window, calls `check_and_preload_active_model()`
- **F8 is excluded** from the hotkey options list (conflicts with the language toggle shortcut)

#### `DashboardWindow` (CTkCTk — main window)
- Positioned in the **bottom-right corner** of the screen (320×240px)
- Contains: header, status card, **language toggle button** (green=RO, blue=EN), 3 action buttons (Settings / Hide / Exit)
- `update_status(state)` — updates the status card text and color
- `update_lang_button(lang)` — updates the language button (called from `toggle_language_callback`)
- Closing via X → `hide_window()` (does not quit the app)

---

### `overlay.py` — floating indicator window

Class: `DictationOverlay`

- **Frameless, transparent, always-on-top** Tkinter window
- **Only visible** while recording or transcribing (hidden in idle via `root.withdraw()`)
- Contains: animated dot, status text, audio visualizer (5 bars), small language toggle button (right side)
- Animation at 25 FPS via `root.after(40, _pulse_animation)`
- All UI updates must be dispatched via `root.after(0, fn)` — thread safety requirement

Key methods:
- `show_listening(lang)` — shows overlay with localized text, pulsing red dot
- `show_transcribing(lang)` — orange dot, wave animation on bars
- `hide()` — hides overlay, resets state to `"idle"`
- `update_language_ui(lang)` — updates language button color and text
- `update_volume(vol)` — updates current volume level (0.0–1.0) for the visualizer

---

### `setup.py` — first-run wizard

- Checks whether the Whisper model exists at `models/whisper-large-v3-turbo-ct2/`
- If missing: downloads and converts the model from HuggingFace using `faster-whisper`
- If present: launches `app.py` directly via `pythonw.exe`
- Writes to `scriba.log` with prefix `[SETUP]`

---

### `run.bat` — launcher

```bat
.venv\Scripts\python.exe setup.py
```
Simple: activates the venv and runs `setup.py`, which decides what to do next.

---

## 4. Full audio pipeline

```
Hotkey/Mouse pressed
  └─> on_trigger_down()
        ├─> app_state.state = "recording"
        ├─> overlay.show_listening(lang)
        └─> start_recording()           # starts sd.InputStream at 16kHz

Hotkey released (or toggle in Press-to-Toggle mode)
  └─> on_trigger_up()
        ├─> app_state.state = "transcribing"
        ├─> overlay.show_transcribing(lang)
        └─> stop_recording_and_get_audio()
              └─> threading.Thread(target=transcribe_and_type_thread, args=(audio_data,))

transcribe_and_type_thread(audio_data):
  ├─> if lang == "ro":
  │     ├─> Write audio_data to _tmp_dictation.wav (16kHz, int16, mono)
  │     ├─> nemo_model.transcribe([tmp_path])
  │     └─> Delete temporary file
  └─> if lang == "en":
        └─> model.transcribe(audio_data, language="en", beam_size=5, vad_filter=True)

  └─> inject_text(text) → clipboard → Ctrl+V → overlay.hide()
```

---

## 5. Configuration (`config.json`)

```json
{
    "hotkey": "Right Control",
    "mouse_button": "None",
    "mode": "Hold to Talk",
    "add_space": true,
    "input_device": "Default",
    "device": "cuda",
    "quantization": "float16",
    "language": "ro"
}
```

- `language`: `"ro"` → NeMo UPB | `"en"` → Whisper
- `mode`: `"Hold to Talk"` | `"Press to Toggle"`
- `device`: `"cuda"` | `"cpu"`
- `quantization`: `"float16"` | `"int8"` (Whisper only)

---

## 6. Known issues and their solutions

### Windows LoaderLock (NeMo + threading)
**Problem:** `import nemo.collections.asr` inside a `threading.Thread` causes a deadlock on Windows (LoaderLock).  
**Solution:** NeMo is imported **exclusively** inside `load_nemo_model()`, which runs on a daemon thread launched from `preload_model_and_listeners` (tray thread). Never import NeMo on the main thread or Tkinter thread.

### NeMo requires a WAV file on disk
**Problem:** `nemo_model.transcribe()` does not accept numpy arrays directly.  
**Solution:** Audio is written to a temporary `_tmp_dictation.wav`, transcribed, then deleted.

### NeMo `transcribe()` returns mixed types
```python
result = nemo_model.transcribe([path])
text = result[0] if isinstance(result, list) else str(result)
if hasattr(text, 'text'):
    text = text.text
text = str(text).strip()
```

### UI freeze on language switch
**Problem:** `import nemo` took ~5s and blocked the Tkinter event loop.  
**Solution:** All NeMo imports removed from `toggle_language_callback()` and `save_and_close()`. Model loading happens exclusively in the background via `check_and_preload_active_model()`.

### Single-instance enforcement
The app uses a **Windows Mutex** (`Global\ScribaRo_SingleInstance_Mutex`) to prevent double launches.

---

## 7. Key dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `faster-whisper` | latest | Optimized Whisper inference |
| `nemo_toolkit[asr]` | 2.7.3 | NeMo UPB model for Romanian |
| `torch` | 2.5.1+cu121 | GPU backend |
| `sounddevice` | latest | Microphone audio capture |
| `pyautogui` + `pyperclip` | latest | Text injection via clipboard |
| `customtkinter` | latest | Modern UI (Settings + Dashboard) |
| `pystray` | latest | System tray icon |
| `pynput` | latest | Keyboard/mouse listeners |
| `Pillow` | latest | Tray icon generation |

---

## 8. Target hardware

- **GPU:** NVIDIA RTX 4070 Laptop (4GB VRAM)
- **CUDA:** 12.1
- **OS:** Windows 11
- Both models run on CUDA — be mindful of VRAM when both are preloaded simultaneously
