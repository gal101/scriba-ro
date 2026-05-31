# 🎤 Scriba Ro

**Offline voice dictation assistant for Windows.**  
Press a key, speak, and the text appears automatically in any application.

---

## What it does

- **Romanian dictation** — uses `upb-nlp/ro-fast-conformer` (NeMo, WER 3.01%)
- **English dictation** — uses `openai/whisper-large-v3-turbo` (CTranslate2)
- **100% offline** — no audio ever leaves your machine
- **Auto-inject** — transcribed text is pasted directly into the active window via clipboard
- **Intelligent post-processing** — automatic punctuation and capitalization restoration for Romanian (using `punctuators` library) and smart spacing injection
- **System tray** — runs quietly in the background, no taskbar clutter
- **Floating overlay** — small visual indicator when recording or transcribing
- **Quick toggle** — switch language with `F8` or via the Dashboard button

## Requirements

- Windows 10/11
- Python 3.10+
- NVIDIA GPU with CUDA (recommended, CPU also works)
- ~13 GB free disk space (models + virtual environment)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/gal101/scriba-ro.git
cd scriba-ro

# 2. Run the setup wizard (downloads models and configures the environment)
run.bat
```

On first run, `launcher.py` automatically checks and installs dependencies, while `setup.py` downloads the Whisper model.  
The NeMo model for Romanian is downloaded separately (see `AGENT.md` for details).

## Usage

1. Run `run.bat`
2. A red icon appears in the system tray and the Dashboard window opens
3. Hold down the configured key (default: **Right Control**) and speak
4. Release the key — the transcribed text appears in the active window
5. Switch language with **F8** or the green/blue button in the Dashboard

## License

Personal project. AI models are subject to their own licenses (Apache 2.0 / CC-BY).
