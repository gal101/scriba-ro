import traceback
import sys
import torch
import numpy as np
import scipy.io.wavfile as wav

def main():
    wav.write('_tmp_test.wav', 16000, np.zeros(16000, dtype=np.float32))
    try:
        import nemo.collections.asr as nemo_asr
        print('Starting restore_from()')
        m = nemo_asr.models.ASRModel.restore_from('models/Speech_To_Text_Finetuning.nemo', map_location=torch.device('cuda'))
        print('Loaded successfully')
        
        print('Starting transcribe()')
        text = m.transcribe(['_tmp_test.wav'], batch_size=1, num_workers=0)
        print('Transcribed successfully:', text)
    except Exception as e:
        traceback.print_exc()

if __name__ == '__main__':
    main()
