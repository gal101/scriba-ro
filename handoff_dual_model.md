# Handoff: Dual-Model ASR Integration — Scriba Ro

## Contextul proiectului

**Scriba Ro** este o aplicație Windows de dictare vocală offline (system tray). Utilizatorul apasă o tastă rapidă, vorbește, iar textul este transcris și injectat automat în fereastra activă via Ctrl+V.

**Locație:** rădăcina repository-ului

---

## Starea curentă a aplicației

### Fișiere principale

| Fișier | Rol |
|--------|-----|
| `app.py` | Logica principală: system tray, audio capture, transcriere, hotkeys, Settings UI |
| `overlay.py` | Fereastra plutitoare transparentă (indicator recording/transcribing) |
| `setup.py` | GUI de setup: descarcă și convertește modelul Whisper la prima rulare |
| `run.bat` | Launcher principal al aplicației |
| `config.json` | Configurație persistată (hotkey, device, limbă etc.) |
| `requirements.txt` | Dependențe pip |
| `test_upb_conformer.py` | Script de test pentru modelul NeMo (nu face parte din app) |
| `TEST_UPB_MODEL.bat` | Lansator dublu-click pentru testul NeMo |

### Modelul Whisper (INSTALAT ✅)

- **Model:** `openai/whisper-large-v3-turbo`
- **Format:** CTranslate2 (faster-whisper)
- **Cale:** `models/whisper-large-v3-turbo-ct2/`
- **Bibliotecă:** `faster-whisper`
- **GPU:** NVIDIA RTX 4070 Laptop, CUDA 12.1, PyTorch 2.5.1+cu121
- **Cuantizare activă:** `float16` pe GPU

### Modelul UPB NeMo (DESCĂRCAT ✅, neintegrat în app)

- **Model:** `upb-nlp/ro-fast-conformer` — cel mai bun ASR offline pentru română (WER 3.01%)
- **Format:** fișier `.nemo` (NVIDIA NeMo toolkit)
- **Cale fișier:** `~/.cache/torch/NeMo/<versiune>/hf_hub_cache/upb-nlp/ro-fast-conformer/<hash>/Speech_To_Text_Finetuning.nemo`
- **Dimensiune:** 453 MB
- **Bibliotecă:** `nemo_toolkit[asr]` v2.7.3 (deja instalat în `.venv`)
- **Încărcare corectă:** `nemo_asr.models.ASRModel.restore_from(nemo_path)`  
  ⚠️ NU `from_pretrained()` — acea metodă caută `model_config.yaml` care nu există

---

## Ce trebuie implementat

### Obiectiv

Utilizatorul vrea să selecteze limba din **Settings UI** (`app.py`) sau din **overlay-ul plutitor** (`overlay.py`), iar selecția să determine **ce model se folosește**:

| Selecție utilizator | Model folosit | Bibliotecă |
|---------------------|---------------|-----------|
| 🇷🇴 **Română** | `upb-nlp/ro-fast-conformer` (NeMo) | `nemo_asr` |
| 🇬🇧 **Engleză** | `openai/whisper-large-v3-turbo` (CTranslate2) | `faster-whisper` |

> **Nu mai este nevoie de "Auto-detect"** — se poate elimina sau păstra neutilizat.

---

## Modificări necesare în `app.py`

### 1. AppState — adaugă starea modelului NeMo

```python
class AppState:
    def __init__(self):
        ...
        self.model = None          # Whisper model (faster-whisper) — pentru engleză
        self.nemo_model = None     # NeMo model — pentru română
        ...
```

### 2. Funcție nouă: `find_nemo_file()`

```python
def find_nemo_file():
    """Gaseste fisierul .nemo in cache-ul NeMo sau in folderul models/."""
    # Mai intai cauta in cache-ul NeMo standard
    nemo_cache = os.path.join(os.path.expanduser("~"), ".cache", "torch", "NeMo")
    for root, dirs, files in os.walk(nemo_cache):
        for f in files:
            if f.endswith(".nemo") and "ro-fast-conformer" in root:
                return os.path.join(root, f)
    # Fallback: cauta in folderul local models/
    local = os.path.join(BASE_DIR, "models", "Speech_To_Text_Finetuning.nemo")
    if os.path.exists(local):
        return local
    return None
```

### 3. Funcție nouă: `load_nemo_model()`

```python
def load_nemo_model():
    if app_state.nemo_model is not None:
        return True
    
    nemo_path = find_nemo_file()
    if not nemo_path:
        log_message("EROARE: Fisierul .nemo nu a fost gasit. Ruleaza TEST_UPB_MODEL.bat pentru a-l descarca.")
        return False
    
    try:
        import nemo.collections.asr as nemo_asr
        log_message(f"Se incarca modelul NeMo din: {os.path.basename(nemo_path)}")
        app_state.nemo_model = nemo_asr.models.ASRModel.restore_from(nemo_path)
        app_state.nemo_model.eval()
        log_message("Modelul NeMo (ro-fast-conformer) incarcat cu succes!")
        return True
    except Exception as e:
        log_message(f"EROARE la incarcarea modelului NeMo: {str(e)}")
        return False
```

### 4. Modifică `transcribe_and_type_thread()` — routing pe baza limbii

```python
def transcribe_and_type_thread(audio_data):
    try:
        lang = app_state.config.get("language", "ro")
        
        if lang == "ro":
            # Foloseste NeMo pentru romana
            if app_state.nemo_model is None:
                success = load_nemo_model()
                if not success:
                    # fallback la Whisper cu language="ro"
                    ...
            
            # NeMo necesita fisier WAV temporar, nu array numpy direct
            import tempfile, wave
            tmp_path = os.path.join(BASE_DIR, "_tmp_dictation.wav")
            audio_int16 = (audio_data * 32767).astype('int16')
            with wave.open(tmp_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_int16.tobytes())
            
            start_time = time.time()
            result = app_state.nemo_model.transcribe([tmp_path])
            text = result[0] if isinstance(result, list) else str(result)
            if hasattr(text, 'text'):
                text = text.text
            text = str(text).strip()
            
            try:
                os.remove(tmp_path)
            except Exception:
                pass
                
        else:
            # Foloseste Whisper pentru engleza (sau alte limbi)
            if app_state.model is None:
                success = load_whisper_model()
                if not success:
                    ...
            
            lang_code = None if lang == "auto" else lang
            segments, info = app_state.model.transcribe(
                audio_data, language=lang_code, beam_size=5, vad_filter=True
            )
            text = "".join([seg.text for seg in segments]).strip()
        
        duration = time.time() - start_time
        log_message(f"Transcriere completa in {duration:.2f}s: '{text}'")
        
        if text:
            inject_text(text, add_space=app_state.config["add_space"])
        else:
            log_message("Niciun cuvant detectat.")
            
    except Exception as e:
        log_message(f"EROARE transcriere: {str(e)}")
    finally:
        app_state.overlay.hide()
        app_state.state = "idle"
```

### 5. Settings UI — simplifica dropdown-ul de limbă

Deja există `self.lang_menu` cu opțiunile `["Română (ro)", "Engleză (en)", "Auto-detect"]`.  
Simplifică la `["Română (ro)", "Engleză (en)"]` — elimină Auto-detect dacă nu mai e relevant.

```python
# In SettingsWindow.__init__:
self.lang_menu = ctk.CTkOptionMenu(
    self.grid_frame,
    values=["Română (ro)", "Engleză (en)"],
    command=self.on_setting_changed
)
lang_map_reverse = {"ro": "Română (ro)", "en": "Engleză (en)"}
self.lang_menu.set(lang_map_reverse.get(app_state.config.get("language", "ro"), "Română (ro)"))
```

### 6. (Opțional dar recomandat) Switch rapid în overlay

`overlay.py` conține `DictationOverlay` — o fereastră tkinter transparentă care apare la recording.  
Se poate adăuga un buton mic `RO / EN` în overlay pentru switch rapid fără a deschide Settings.

---

## Arhitectura curentă a fluxului audio

```
Hotkey apăsat
  └─> start_recording()          # porneste sd.InputStream la 16kHz
        └─> audio_callback()     # colecteaza chunks in audio_queue
Hotkey eliberat / toggle
  └─> stop_recording_and_get_audio()   # concateneaza chunks, resampleaza daca necesar
        └─> transcribe_and_type_thread(audio_data)  # pe thread separat
              └─> [NeMo sau Whisper] -> inject_text()
```

---

## Dependențe relevante

```
faster-whisper       # pentru Whisper (engleza)
nemo_toolkit[asr]    # v2.7.3, deja instalat in .venv
sounddevice          # captare audio
pyautogui + pyperclip # injectare text
customtkinter        # UI Settings + Dashboard
pystray              # system tray icon
```

---

## Atenție la NeMo pe Windows

- NeMo are warning despre `ffmpeg` lipsă — nu e necesar pentru inferență simplă, se poate ignora
- NeMo afișează multe log-uri la încărcare (`[NeMo I/W ...]`) — normal, nu sunt erori
- NeMo `transcribe()` returnează o listă — elementele pot fi string sau obiect cu `.text`
- NeMo are nevoie de **fișier WAV** pe disc, nu de array numpy direct
- La prima încărcare a modelului, NeMo poate dura 10-20 secunde (inițializare CUDA)

---

## Starea config.json curentă

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

Valoarea `"language"` trebuie să fie `"ro"` pentru NeMo sau `"en"` pentru Whisper.
