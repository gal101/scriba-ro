import traceback
import sys
import numpy as np
import scipy.io.wavfile as wav

wav.write('_tmp_test.wav', 16000, np.zeros(16000, dtype=np.int16))
try:
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.EncDecRNNTBPEModel.restore_from('models/Speech_To_Text_Finetuning.nemo')
    print('Loaded successfully')
    m.transcribe(['_tmp_test.wav'])
    print('Transcribed successfully')
except Exception as e:
    traceback.print_exc()
