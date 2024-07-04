# import nemo.collections.asr as nemo_asr
import sys
import os
sys.path.append(os.path.abspath('/workspace/nemo/NeMo-opensource/nemo/collections'))
sys.path.append(os.path.abspath('/workspace/nemo/NeMo-opensource/nemo'))
import asr as nemo_asr

def load_model(model_name):
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    return model

model = load_model("stt_en_conformer_ctc_large")
model.transcribe(["/disk1/mixed_dataset_000/mixed_audios/train/08zhZZn29jc_496fe_1997_Peters_Township_High_School_Commencement_SLASH_1997_Peters_Township_High_School_Commencement_DOT_mp3_00010.wav"])