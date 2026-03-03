# pip install git+https://github.com/openai/whisper.git
# pip install torchaudio

import whisper # OpenAI
from pathlib import Path
from jiwer import wer
import warnings
warnings.simplefilter('ignore')

from src.dataset import LibriDataset_List
    
if __name__ == "__main__":

    model = whisper.load_model("tiny")

    DATASET_ROOT = "C:/Users/HuyenDT/Downloads/LibriSpeech"
    list = LibriDataset_List(DATASET_ROOT, ["dev-clean"])

    FILE_INDEX = 0 
    audio_path, org_transcript = list.librispeech_data[FILE_INDEX]
    audio_path, org_transcript 

    audio_file = Path(audio_path)
    result = model.transcribe(str(audio_file))
    
    pred_transcript = str(result["text"]).upper()

    print("Transcribed text:", pred_transcript)
    print("Original text:", org_transcript)
    print("WER:", wer(org_transcript, pred_transcript))
