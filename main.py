from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import io 
import torch
import torchaudio
import logging
import uuid
import numpy as np
from src.inference import inference2
from src.model import DeepSpeech2


app = FastAPI() 
logger = logging.getLogger(__name__)
SAMPLING_RATE = 16000
WAV2VEC2_FINNISH = "jonatasgrosman/wav2vec2-large-xlsr-53-finnish"
WAV2VEC2_BASE = "facebook/wav2vec2-base-960h"

processor_en = Wav2Vec2Processor.from_pretrained(WAV2VEC2_BASE) # wav2vec2-large/base-960h
model_en = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_BASE)

processor_fi = Wav2Vec2Processor.from_pretrained(WAV2VEC2_FINNISH)
model_fi = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_FINNISH)

model2 = DeepSpeech2(rnn_hidden_size= 128,rnn_depth = 2 )
model2.load_state_dict(torch.load("best_weights.pt",weights_only=True)) 
model2.eval() 
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")

MODEL_PATH = "./models/wav2vec2"

# model.save_pretrained(MODEL_PATH)
# processor.save_pretrained(MODEL_PATH)
# processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH) # wav2vec2-base-960h
# model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)

# Cho phép domain frontend truy cập
origins = [
    "http://localhost",
    "http://localhost:3000",  # domain frontend của bạn
    "https://my-frontend.com",  # nếu có deploy
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # hoặc ["*"] để cho tất cả
    allow_credentials=False,
    allow_methods=["*"],     # GET, POST, PUT, DELETE…
    allow_headers=["*"],     # header tuỳ ý
)

@app.get("/")
def root():
    return "Hello, I am alive"

@app.post("/predict")
async def predict(file: UploadFile, la: str = Form(...)):
    try:
        print(la)
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise ValueError("Empty audio file")
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes), normalize=True)
        if sr != SAMPLING_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLING_RATE) # re-sample to 16.000

        waveform = waveform.flatten()
        duration = np.floor(len(waveform) / sr)

        processor = processor_fi if la=="fi" else processor_en
        model = model_fi if la=="fi" else model_en

        inputs = processor(waveform, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits # ** from dictionary to keyword arguments, in case too many args => no need to pass one-by-one
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)

        return {
            "transcript": transcription[0],
            "filename": file.filename,
            "duration": duration
        }
    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.exception("Model inference failed")
        raise HTTPException(status_code=500, detail="Model inference error")

@app.post("/predict2")
async def predict2(file: UploadFile, la: str):
    try:
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise ValueError("Empty audio file")
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes), normalize=True)
        waveform = waveform.flatten()
        duration = np.floor(len(waveform)/sr)

        pred_transcript = inference2(audio_bytes, model2, tokenizer)

        return {
            "transcript": pred_transcript,
            "filename": file.filename,
            "duration": duration
        }
    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.exception("Model inference failed")
        raise HTTPException(status_code=500, detail="Model inference error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)