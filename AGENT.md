# AGENT.md — Ghid pentru agenți AI care lucrează la Scriba Ro

Acest document descrie în detaliu arhitectura, fluxul de date, fișierele și deciziile tehnice
ale proiectului **Scriba Ro**. Citește-l integral înainte de a face orice modificare.

---

## 1. Ce este aplicația

Scriba Ro este o aplicație Windows de **dictare vocală offline** cu system tray.
Utilizatorul ține apăsată o tastă rapidă, vorbește, iar textul transcris este injectat
automat în fereastra activă via clipboard (Ctrl+V simulat).

Există **două motoare ASR**, selectate în funcție de limbă:

| Limbă | Model | Bibliotecă | Fișier |
|-------|-------|------------|--------|
| 🇷🇴 Română | `upb-nlp/ro-fast-conformer` | NeMo 2.7.3 | `models/Speech_To_Text_Finetuning.nemo` |
| 🇬🇧 Engleză | `openai/whisper-large-v3-turbo` | faster-whisper | `models/whisper-large-v3-turbo-ct2/` |

---

## 2. Structura fișierelor

```
Whisper/
├── app.py                  # Logica principală (entry point)
├── overlay.py              # Fereastra plutitoare (Tkinter)
├── setup.py                # Wizard de setup la prima rulare
├── run.bat                 # Launcher: rulează setup.py, apoi app.py
├── requirements.txt        # Dependențe pip
├── config.json             # Configurație persistată (gitignored)
├── scriba.log              # Log runtime (gitignored)
├── models/
│   ├── whisper-large-v3-turbo-ct2/   # Model Whisper (CTranslate2)
│   └── Speech_To_Text_Finetuning.nemo # Model NeMo UPB (română)
└── .venv/                  # Mediu virtual Python (gitignored)
```

---

## 3. Fișierele principale

### `app.py` — logica centrală

Conține toate componentele aplicației:

#### `AppState` (clasă globală singleton)
```python
app_state = AppState()
```
Stochează starea globală:
- `app_state.model` — instanța faster-whisper (Whisper, pentru EN)
- `app_state.nemo_model` — instanța NeMo ASRModel (pentru RO)
- `app_state.config` — dict cu configurația curentă
- `app_state.state` — `"idle"` / `"recording"` / `"transcribing"`
- `app_state.overlay` — instanța `DictationOverlay`
- `app_state.dashboard` — instanța `DashboardWindow`
- `app_state.tray_icon` — instanța pystray
- `app_state.keyboard_listener`, `app_state.mouse_listener`

#### Funcții critice

| Funcție | Rol |
|---------|-----|
| `load_config()` / `save_config()` | Citește/scrie `config.json` |
| `find_nemo_file()` | Caută fișierul `.nemo` în `models/` sau `~/.cache/torch/NeMo/` |
| `load_nemo_model()` | Încarcă modelul NeMo pe CUDA/CPU, setează `app_state.nemo_model` |
| `load_whisper_model()` | Încarcă modelul Whisper, setează `app_state.model` |
| `check_and_preload_active_model()` | Detectează limba activă și lansează încărcarea modelului corect pe un daemon thread |
| `transcribe_and_type_thread(audio_data)` | Rutează transcrierea la NeMo (RO) sau Whisper (EN), injectează textul |
| `start_recording()` | Pornește `sd.InputStream` la 16kHz |
| `stop_recording_and_get_audio()` | Colectează chunks din `audio_queue`, resamplează dacă necesar |
| `inject_text(text, add_space)` | Copiază în clipboard și simulează Ctrl+V |
| `toggle_language_callback()` | Comută limba RO↔EN, salvează config, preîncarcă modelul nou, actualizează toate UI-urile |
| `restart_listeners()` | Repornește keyboard/mouse listeners după schimbarea hotkey-ului |
| `preload_model_and_listeners(icon)` | Rulează pe tray thread după pornire: inițializează listeners și pornește preîncărcarea |
| `main()` | Entry point: single-instance check, creare tray icon, pornire dashboard, mainloop |

#### `SettingsWindow` (CTkToplevel)
- Dropdown limbă mapează `"🇷🇴 Română — NeMo UPB"` ↔ `"ro"` și `"🇬🇧 Engleză — Whisper"` ↔ `"en"`
- `save_and_close()` salvează config, repornește listeners, distruge fereastra, apelează `check_and_preload_active_model()`
- **F8 este exclus** din lista de hotkey-uri (conflict cu toggle-ul de limbă)

#### `DashboardWindow` (CTkCTk — fereastra principală)
- Poziționată în **colțul jos-dreapta** al ecranului (320×240px)
- Conține: header, card de status, **buton de toggle limbă** (verde=RO, albastru=EN), 3 butoane (Setări/Ascunde/Ieșire)
- `update_status(state)` — actualizează textul și culoarea cardului de status
- `update_lang_button(lang)` — actualizează butonul de limbă (chemat din `toggle_language_callback`)
- Minimizare la X → `hide_window()` (nu iese din aplicație)

---

### `overlay.py` — fereastra plutitoare

Clasă: `DictationOverlay`

- Fereastră **frameless, transparentă, always-on-top** (Tkinter)
- **Vizibilă DOAR** când înregistrezi sau transcrii (ascunsă în idle cu `root.withdraw()`)
- Conține: dot animat, text status, vizualizator audio (5 bare), buton toggle limbă (mic, în dreapta)
- Animație la 25 FPS via `root.after(40, _pulse_animation)`
- Toate update-urile UI trebuie făcute via `root.after(0, fn)` — thread-safe

Metode importante:
- `show_listening(lang)` — arată overlay cu text localizat, dot roșu pulsant
- `show_transcribing(lang)` — dot portocaliu, animație wave pe bare
- `hide()` — ascunde overlay, resetează state la `"idle"`
- `update_language_ui(lang)` — actualizează culoarea și textul butonului de limbă
- `update_volume(vol)` — actualizează volumul curent (0.0–1.0) pentru vizualizator

---

### `setup.py` — wizard de primă rulare

- Verifică dacă există modelul Whisper în `models/whisper-large-v3-turbo-ct2/`
- Dacă nu există: descarcă și convertește modelul din HuggingFace cu `faster-whisper`
- Dacă există: lansează direct `app.py` via `pythonw.exe`
- Scrie în `scriba.log` cu prefixul `[SETUP]`

---

### `run.bat` — launcher

```bat
.venv\Scripts\python.exe setup.py
```
Simplu: activează venv-ul implicit și rulează setup.py care decide ce face mai departe.

---

## 4. Fluxul audio complet

```
Hotkey/Mouse apăsat
  └─> on_trigger_down()
        ├─> app_state.state = "recording"
        ├─> overlay.show_listening(lang)
        └─> start_recording()           # pornește sd.InputStream la 16kHz

Hotkey eliberat (sau toggle în modul Press-to-Toggle)
  └─> on_trigger_up()
        ├─> app_state.state = "transcribing"
        ├─> overlay.show_transcribing(lang)
        └─> stop_recording_and_get_audio()
              └─> threading.Thread(target=transcribe_and_type_thread, args=(audio_data,))

transcribe_and_type_thread(audio_data):
  ├─> dacă lang == "ro":
  │     ├─> Scrie audio_data în _tmp_dictation.wav (16kHz, int16, mono)
  │     ├─> nemo_model.transcribe([tmp_path])
  │     └─> Șterge fișierul temporar
  └─> dacă lang == "en":
        └─> model.transcribe(audio_data, language="en", beam_size=5, vad_filter=True)

  └─> inject_text(text) → clipboard → Ctrl+V → overlay.hide()
```

---

## 5. Configurație (`config.json`)

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
- `quantization`: `"float16"` | `"int8"` (doar pentru Whisper)

---

## 6. Probleme cunoscute și soluțiile lor

### LoaderLock pe Windows (NeMo + threading)
**Problemă:** `import nemo.collections.asr` din interiorul unui `threading.Thread` cauzează deadlock pe Windows (LoaderLock).  
**Soluție:** Importul NeMo se face **exclusiv** în `load_nemo_model()` care rulează pe un daemon thread lansat din `preload_model_and_listeners` (tray thread). Nu face niciodată `import nemo` pe main thread sau pe Tkinter thread.

### NeMo necesită fișier WAV pe disc
**Problemă:** `nemo_model.transcribe()` nu acceptă numpy arrays direct.  
**Soluție:** Audio-ul e scris temporar în `_tmp_dictation.wav`, transcris, apoi șters.

### NeMo `transcribe()` returnează tipuri variate
```python
result = nemo_model.transcribe([path])
text = result[0] if isinstance(result, list) else str(result)
if hasattr(text, 'text'):
    text = text.text
text = str(text).strip()
```

### Freeze UI la schimbarea limbii
**Problemă:** `import nemo` dura ~5s și bloca Tkinter.  
**Soluție:** Toate importurile NeMo sunt eliminate din `toggle_language_callback()` și `save_and_close()`. Modelul se încarcă **doar** în background via `check_and_preload_active_model()`.

### Single-instance
Aplicația folosește un **Windows Mutex** (`Global\ScribaRo_SingleInstance_Mutex`) pentru a preveni rularea dublă.

---

## 7. Dependențe cheie

| Pachet | Versiune | Rol |
|--------|----------|-----|
| `faster-whisper` | latest | Inferență Whisper optimizată |
| `nemo_toolkit[asr]` | 2.7.3 | Model NeMo UPB pentru română |
| `torch` | 2.5.1+cu121 | Backend GPU |
| `sounddevice` | latest | Captare audio microfon |
| `pyautogui` + `pyperclip` | latest | Injectare text via clipboard |
| `customtkinter` | latest | UI modern (Settings + Dashboard) |
| `pystray` | latest | System tray icon |
| `pynput` | latest | Keyboard/mouse listeners |
| `Pillow` | latest | Generare icon tray |

---

## 8. Hardware țintă

- **GPU:** NVIDIA RTX 4070 Laptop (4GB VRAM)
- **CUDA:** 12.1
- **OS:** Windows 11
- Ambele modele rulează pe CUDA simultan — atenție la VRAM când sunt preîncărcate în același timp
