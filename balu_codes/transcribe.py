# import nemo.collections.asr as nemo_asr
import sys
import os
sys.path.append(os.path.abspath('/home/bld56/gsoc/nemo/NeMo-opensource'))
import nemo.collections.asr as nemo_asr

def load_model(model_name):
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    return model
model = load_model("stt_en_conformer_ctc_large")
# model = load_model("QuartzNet15x5Base-En")
model.transcribe(["/disk1/it1/mixed_audios/009LTXtP4vE_c053b1_171114BCPC_SLASH_171114-BC-PC_DOT_mp3_00035.wav"])