import traceback
import sys
import os
try:
    import nemo.collections.asr as nemo_asr
    print("nemo imported")
    nemo_asr.models.EncDecRNNTBPEModel.restore_from('models/Speech_To_Text_Finetuning.nemo')
    print('SUCCESS')
except Exception as e:
    with open('error_trace.txt', 'w') as f:
        traceback.print_exc(file=f)
    print("Caught exception! See error_trace.txt")
except BaseException as be:
    with open('error_trace_base.txt', 'w') as f:
        traceback.print_exc(file=f)
    print("Caught base exception!")
