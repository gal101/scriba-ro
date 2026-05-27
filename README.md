# 🎤 Scriba Ro

**Asistent local de dictare vocală offline pentru Windows.**  
Apasă o tastă, vorbești, textul apare automat în orice aplicație.

---

## Ce face

- **Dictare în Română** — folosește modelul `upb-nlp/ro-fast-conformer` (NeMo, WER 3.01%)
- **Dictare în Engleză** — folosește `openai/whisper-large-v3-turbo` (CTranslate2)
- **100% offline** — niciun audio nu pleacă în cloud
- **Injectare automată** — textul e lipit direct în fereastra activă via clipboard
- **System tray** — rulează discret în fundal, fără ferestre în taskbar
- **Overlay plutitor** — indicator vizual mic când înregistrezi sau transcrii
- **Toggle rapid** — schimbi limba cu `F8` sau din butonul din Dashboard

## Cerințe sistem

- Windows 10/11
- Python 3.10+
- NVIDIA GPU cu CUDA (recomandat, funcționează și pe CPU)
- ~13 GB spațiu liber (modele + mediu virtual)

## Instalare

```bash
# 1. Clonează repo-ul
git clone https://github.com/gal101/scriba-ro.git
cd scriba-ro

# 2. Rulează setup-ul (descarcă modelele și configurează mediul)
run.bat
```

La prima rulare, `setup.py` creează automat mediul virtual și descarcă modelul Whisper.  
Modelul NeMo pentru română trebuie descărcat separat (instrucțiuni în `AGENT.md`).

## Utilizare

1. Rulează `run.bat`
2. Apare iconița roșie în system tray și fereastra Dashboard
3. Ține apăsată tasta configurată (implicit: **Right Control**) și vorbești
4. Eliberezi tasta — textul apare în fereastra activă
5. Schimbi limba cu **F8** sau din butonul verde/albastru din Dashboard

## Licență

Proiect personal. Modelele AI au licențele lor proprii (Apache 2.0 / CC-BY).
