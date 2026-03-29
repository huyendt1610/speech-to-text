from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import io 
import torch
import torchaudio
import torchaudio.functional as F
import logging
import uuid
import numpy as np
from src.inference import inference2
from src.model import DeepSpeech2
import whisper # OpenAI

app = FastAPI() 
logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000
WAV2VEC2_FINNISH = "jonatasgrosman/wav2vec2-large-xlsr-53-finnish"
WAV2VEC2_BASE = "facebook/wav2vec2-base-960h"

processor_en = Wav2Vec2Processor.from_pretrained(WAV2VEC2_BASE) # wav2vec2-large/base-960h
model_en = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_BASE)

processor_fi = Wav2Vec2Processor.from_pretrained(WAV2VEC2_FINNISH)
model_fi = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_FINNISH)

deepSpeech2_model = DeepSpeech2(rnn_hidden_size= 128,rnn_depth = 2 )
deepSpeech2_model.load_state_dict(torch.load("best_weights.pt",weights_only=True)) 
deepSpeech2_model.eval() 
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")

whisper_model = whisper.load_model("tiny")
MODEL_PATH = "./models/wav2vec2"

# model.save_pretrained(MODEL_PATH)
# processor.save_pretrained(MODEL_PATH)
# processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH) # wav2vec2-base-960h
# model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)

# Allow domain frontend access
origins = [
    "http://localhost",
    "http://localhost:3000",  # frontend domain
    "https://my-frontend.com",  # if deployed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for all with no striction 
    allow_credentials=False,
    allow_methods=["*"],     # GET, POST, PUT, DELETE…
    allow_headers=["*"],     # header options
)

@app.get("/")
def root():
    return "Hello, I am alive"

@app.post("/predict")
async def predict(file: UploadFile, la: str = Form(...), model_name: str = Form(...)):
    try:
        audio_bytes = await file.read()
        waveform, duration = validateFile(audio_bytes)

        language = "en"
        print(language)
        if (model_name == "deepspeech2"): 
            pred_transcript = inference2(waveform, deepSpeech2_model, tokenizer)
        elif (model_name == "whisper"): 
            waveform = waveform.flatten()
            result = whisper_model.transcribe(waveform, language= la) if la != "au" else whisper_model.transcribe(waveform)
            pred_transcript = result["text"]
            language =result["language"]
        else: 
            waveform = waveform.flatten()
            # processor = processor_fi if la=="fi" else processor_en
            # model = model_fi if la=="fi" else model_en
            processor, model, language = load_model(model_name)

            chunks = chunk_audio(waveform, chunk_sec=10, overlap_sec=1)
            texts = transcribe_chunks(processor, model, chunks)
            pred_transcript = " ".join(texts)

        return {
            "transcript": pred_transcript,
            "filename": file.filename,
            "duration": duration,
            "language": language
        }
    except Exception as e:
        logger.exception("Model inference failed")
        print(e)
        raise HTTPException(status_code=500, detail="Model inference error")

def load_model(model_name):
    match model_name:
        case "wav2vec2":
            return processor_en, model_en, "en"
        case "wav2vec2_fi":
            return processor_fi, model_fi, "fi"
        case "whisper":
            return "Load deepspeech"
        case "deepspeech2": 
            return 
        case _:
            raise ValueError("Unknown model")
        
def validateFile(audio_bytes):  
    if len(audio_bytes) == 0:
            raise ValueError("Empty audio file")
        
    waveform, orig_sr = torchaudio.load(io.BytesIO(audio_bytes), normalize=True)
    duration = np.floor(waveform.shape[1]/orig_sr)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)       # convert to mono

    if orig_sr != SAMPLING_RATE:
        #print(f"Changed the sampling rate from {orig_sr}: {SAMPLING_RATE}")
        waveform = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=SAMPLING_RATE) # re-sample to 16.000

    # trim silence
    segments , _ = F.detect_speech(waveform, SAMPLING_RATE) # return [(start1, end1), (start2, end2), ...]

    start = segments[0][0]
    end = segments[-1][1]
    waveform = waveform[:, start:end] # trim start, end of audio 

    return waveform, duration 

def transcribe_chunks(processor, model, chunks, sr =16000, device="cpu"):
    results = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")

        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(predicted_ids)[0]

        results.append(text)

    return results

def chunk_audio(audio, sr=16000, chunk_sec=10, overlap_sec=1):
    chunk_size = int(chunk_sec * sr)
    hop_size = int((chunk_sec - overlap_sec) * sr)

    chunks = []
    for start in range(0, len(audio), hop_size):
        end = start + chunk_size
        chunk = audio[start:end]
        print(start, end)

        if len(chunk) > 0:
            chunks.append(chunk)

    return chunks

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)