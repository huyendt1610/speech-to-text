import torch 
# import whisper
import torchaudio.transforms as T
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from optimum.onnxruntime import ORTModelForCTC
from app.config import settings
from app.utils.audio import chunk_audio, transcribe_chunks
from shared.models.deepspeech2 import DeepSpeech2
import json 
from faster_whisper import WhisperModel 

class DeepSpeech2Inference: 
    def __init__(self, model_path: str, config_path: str):
        with open(config_path) as f:
            config = json.load(f)
        
        model = DeepSpeech2(rnn_hidden_size=config["rnn_hidden_size"],rnn_depth = config["rnn_depth"] )
        model.load_state_dict(torch.load(model_path,weights_only=True)) 
        model.eval() 

        # Optimization
        model = torch.compile(model) # compile to graph, then quantize

        # torch.onnx.export(model, )

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
        # self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(settings.DEVICE)

        # optimize using onnxruntime
        # export=True → automatically export ONNX first time, then cache
        # provider = .to(device)
        provider = "CUDAExecutionProvider" if settings.DEVICE == "cuda" else "CPUExecutionProvider"
        self.model = ORTModelForCTC.from_pretrained(model_name, export= True, provider=provider)

    def transcribe(self, waveform, language): 
        waveform = waveform.flatten()
        chunks = chunk_audio(waveform)
        # ORT get CPU tensor => automatically copy to GPU via CUDAExecutionProvider
        texts = transcribe_chunks(self.processor, self.model, chunks, settings.SAMPLING_RATE, 'cpu' ) #settings.DEVICE
        return " ".join(texts), language
    
class WhisperInference: 
    def __init__(self, model_name: str):
        # self.model = whisper.load_model(model_name, device=settings.DEVICE)
        self.model = WhisperModel('medium', device=settings.DEVICE, compute_type='int8' if settings.DEVICE == 'cpu' else 'float16')

    def transcribe(self, waveform, la):
        # waveform = waveform.flatten()

        # result = self.model.transcribe(waveform, language=la) if la != "au" else self.model.transcribe(waveform)
        # return result["text"], result["language"]

        waveform = waveform.flatten().numpy()  # faster-whisper cần numpy array
        language = la if la != "au" else None  # None = auto detect
        segments, info = self.model.transcribe(waveform, language=language, beam_size=1)
        text = " ".join([s.text for s in segments])
        return text, info.language