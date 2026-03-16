from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import io 
import torch
import torchaudio
import logging
import uuid
import numpy as np


app = FastAPI() 
logger = logging.getLogger(__name__)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h") # wav2vec2-large/base-960h
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

MODEL_PATH = "./models/wav2vec2"

# model.save_pretrained(MODEL_PATH)
# processor.save_pretrained(MODEL_PATH)
# processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH) # wav2vec2-base-960h
# model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)

# Cho phép domain frontend truy cập
origins = [
    "http://localhost:3000",  # domain frontend của bạn
    "https://my-frontend.com",  # nếu có deploy
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] ,  # hoặc ["*"] để cho tất cả
    allow_credentials=False,
    allow_methods=["*"],     # GET, POST, PUT, DELETE…
    allow_headers=["*"],     # header tuỳ ý
)

@app.get("/")
def root():
    return "Hello, I am alive"

@app.post("/predict")
async def predict(file: UploadFile):
    try:

        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise ValueError("Empty audio file")
        # return {"filename": file.filename, "size": len(audio_bytes)}
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes), normalize=True)
        waveform = waveform.flatten()

        duration = np.floor(len(waveform) / sr)

        inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)