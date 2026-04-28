from app.model_loader import get_model
from app.utils.audio import validateFile

def run_inference(model_name, language, audio_bytes):
    waveform, duration = validateFile(audio_bytes)
    model = get_model(model_name) 
    pred_transcript, language = model.transcribe(waveform, language)
    return pred_transcript, language, duration
