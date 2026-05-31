# Hand-off Document

## Original Task
The user requested two things initially:
1. Fix a bug where the string `<unk>` appears in the transcribed Romanian text. It should be replaced with a space (`" "`) during post-processing.
2. The user also reported that text injection completely stopped working after recording for the Romanian model, while the English model still works fine.

## The Blocking Issue (Current State)
While attempting to investigate, we discovered that the application crashes completely and silently (exit code 1) when trying to load the Romanian ASR model via `ASRModel.restore_from()`.

### What caused it?
Based on git history and logs, on May 27th (commit `e485e47`), the `punctuators` package was introduced. Installing `punctuators` caused `pip` to automatically update `transformers` (from `4.35.0` or similar to `4.51.3+`), `pytorch-lightning`, and ultimately break compatibility between `nemo_toolkit[asr]`, `transformers`, and `torch` (CUDA support). 

### What was attempted to fix it?
1. Created `test_transcribe_cuda.py` to isolate the model loading outside the GUI.
2. Noticed `pip` kept uninstalling `torch==2.5.1+cu121` and replacing it with incompatible/CPU versions (e.g. `2.12.0` CPU) due to dependency conflicts.
3. Manually reinstalled `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`, `torchaudio==2.5.1+cu121`.
4. Manually installed `transformers==4.57.6`, `pytorch-lightning==2.4.0`, and `nemo_toolkit[asr]==2.7.3` using `--no-deps` to prevent them from destroying the PyTorch installation.
5. Fixed `tokenizers` dependency by forcing `tokenizers==0.22.2`.

### The Result
Even with the correct packages seemingly installed, `python test_transcribe_cuda.py` STILL fails silently with `exit code: 1`. It prints the initial NeMo startup warnings and then crashes the python process native-side.

## Next Agent Action Plan

**1. Debug the Silent Crash (Exit Code 1)**
- The crash happens natively in C++/CUDA. You can try running `python -X faulthandler test_transcribe_cuda.py` to see where the segfault happens.
- It is highly likely that there is still a library mismatch, possibly `torchaudio` conflicting with the Windows CUDA drivers, or `transformers` tokenizer bindings crashing.
- *Alternative strategy*: Recreate the `.venv` entirely using the known good state before May 27th, and then carefully add `punctuators` without updating `nemo_toolkit` or `transformers`. 

**2. Fix the `<unk>` Token (Post-processing)**
- Once the model actually loads and runs without crashing, find the post-processing pipeline (likely in `app.py`, unde rezultatul ASR este procesat).
- Înlocuiește tokenul `<unk>` cu cratimă (`-`). Codul ar trebui să fie: `text = text.replace("<unk>", "-")`. Acest lucru se întâmplă deoarece modelul produce `<unk>` la cuvintele care au cratimă în limba română.

**3. Fix Text Injection for Romanian**
- Investigate why the text injection doesn't happen anymore for the Romanian model. If the crash is fixed, this might magically be fixed too (because the crash was stopping the thread). If it doesn't fix it, trace the signals from the ASR thread to the GUI injection thread.

## Useful Files
- `./test_transcribe_cuda.py` (Use this for fast iteration on the model loading without GUI overhead).
- `./app.py` (Main application).
- `./scriba.log` (Application logs).
