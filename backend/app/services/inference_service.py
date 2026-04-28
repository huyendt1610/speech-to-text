import torch 
import whisper
import torchaudio.transforms as T
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from app.config import settings
from app.utils.audio import chunk_audio, transcribe_chunks
from shared.models.deepspeech2 import DeepSpeech2
import json 


class DeepSpeech2Inference: 
    def __init__(self, model_path: str, config_path: str):
        with open(config_path) as f:
            config = json.load(f)
        
        model = DeepSpeech2(rnn_hidden_size=config["rnn_hidden_size"],rnn_depth = config["rnn_depth"] )
        model.load_state_dict(torch.load(model_path,weights_only=True)) 
        model.eval() 

        self.device = settings.DEVICE

        self.model = model.to(self.device)
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")

        self.audio2mels = T.MelSpectrogram(
                sample_rate = settings.SAMPLING_RATE,
                n_mels=80 
            )

        self.amp2db = T.AmplitudeToDB(
            top_db=80.0
        ) 

    def transcribe(self, waveform, language = "en"): 
        # audio: waveform
        mel = self.audio2mels(waveform) 
        mel = self.amp2db(mel) 
        mel = (mel - mel.mean())/(mel.std() + 1e-6) # 1e-6 to avoid deviding by zero, to nomalize
        mel = mel.unsqueeze(0) # add in batch dimension 
        # print(mel.shape)

        src_len = torch.tensor([mel.shape[-1]]).to(self.device) # compute src_len

        with torch.no_grad():
            pred_logits, _ = self.model(mel.to(self.device), src_len)

        pred_tokens = pred_logits.squeeze().argmax(axis=-1).tolist() 

        pred_transcript = self.tokenizer.decode(pred_tokens)

        return pred_transcript, language
        
class Wav2Vec2Inference: 
    def __init__(self, model_name: str):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name) # wav2vec2-large/base-960h
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(settings.DEVICE)

    def transcribe(self, waveform, language): 
        waveform = waveform.flatten()
        chunks = chunk_audio(waveform)
        texts = transcribe_chunks(self.processor, self.model, chunks, settings.SAMPLING_RATE, settings.DEVICE)
        return " ".join(texts), language
    
class WhisperInference: 
    def __init__(self, model_name: str):
        self.model = whisper.load_model(model_name, device=settings.DEVICE)

    def transcribe(self, waveform, la):
        waveform = waveform.flatten()

        result = self.model.transcribe(waveform, language=la) if la != "au" else self.model.transcribe(waveform)
        return result["text"], result["language"]