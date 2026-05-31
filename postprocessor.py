import re

# Modelul va fi încărcat lenes (lazy load) pentru a nu încetini pornirea aplicației
_ro_model = None

def load_ro_model():
    global _ro_model
    if _ro_model is None:
        try:
            from punctuators.models import PunctCapSegModelONNX
            print("[PostProcessor] Se incarcă modelul de punctuație pcs_romance (ONNX)...")
            _ro_model = PunctCapSegModelONNX.from_pretrained("pcs_romance")
            print("[PostProcessor] Modelul de punctuație a fost încărcat cu succes.")
        except ImportError:
            print("[PostProcessor] Eroare: Pachetul 'punctuators' nu este instalat.")
        except Exception as e:
            print(f"[PostProcessor] Eroare la incarcarea modelului pcs_romance: {e}")

def process_english(text):
    if not text:
        return text
    # Înlocuim "slash" cu "/"
    text = re.sub(r"(?i)\bslash\b", "/", text)
    return text

def process_romanian(text):
    if not text:
        return text
        
    global _ro_model
    if _ro_model is None:
        load_ro_model()
        
    if _ro_model is not None:
        try:
            # model.infer primește o listă de texte și returnează o listă de liste
            results = _ro_model.infer([text])
            if results and len(results) > 0:
                res = results[0]
                if isinstance(res, list) and len(res) > 0:
                    final_text = " ".join(res)
                elif isinstance(res, str):
                    final_text = res
                else:
                    return text
                    
                # Modelul pcs_romance pare să înlocuiască cratima (-) cu <unk> în cuvinte compuse (ex: "să-ți", "vi-n")
                final_text = re.sub(r"(?i)<unk>", "-", final_text).replace("  ", " ").strip()
                return final_text
        except Exception as e:
            print(f"[PostProcessor] Eroare in timpul aplicarii punctuatiei: {e}")
            
    return text
