
from app.services.inference_service import DeepSpeech2Inference, Wav2Vec2Inference, WhisperInference
from app.config import settings

MODEL_REGISTRY = {
    "wav2vec2": Wav2Vec2Inference(settings.WAV2VEC2_BASE), 
    "wav2vec2_fi": Wav2Vec2Inference(settings.WAV2VEC2_FINNISH),
    "whisper": WhisperInference(settings.WHISPER_MODEL_NAME),
    "deepspeech2": DeepSpeech2Inference(settings.DEEPSPEECH2_MODEL_PATH, settings.DEEPSPEECH2_CONFIG_PATH)
}

def get_model(model_name):
    return MODEL_REGISTRY.get(model_name)
