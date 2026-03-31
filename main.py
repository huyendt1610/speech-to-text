from fastapi import FastAPI, UploadFile, File, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import io 
import torch
import torchaudio
from torchaudio.transforms import Vad
import logging
import uuid
import numpy as np
from src.inference import inference2
from src.model import DeepSpeech2
import whisper # OpenAI
import json 
import subprocess

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

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        model_name = ""
        la = "en"
        while True:
            msg = await ws.receive()
            if "text" in msg: # JSON 
                data = json.loads(msg["text"])
                msg_type = data.get("type")

                if msg_type == "start":
                    model_name = data.get("model_name")
                    print("Start config:", model_name)
                    la = data.get("la")

                elif msg_type == "stop":
                    print("Stop stream")
                    break

            # 🔹 2. Nếu là audio binary
            elif "bytes" in msg:
                
                audio_chunk = msg["bytes"]

                # xử lý audio ở đây
                # print(model_name, la, audio_chunk)
                audio_np = decode_webm_chunk(audio_chunk)
                print(audio_np)
                # a , _, _  = inferenceText(model_name, la, audio_chunk)
                # print(a)
                text = "hello"

                await ws.send_json({ # await ws.send_text(text)
                    "type": "transcript",
                    "text": text,
                    "is_final": False
                })

    except WebSocketDisconnect:
        print("Client disconnected")

    finally:
        print("Cleanup here")

def inferenceText(model_name, la, audio_bytes):
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

    return pred_transcript, language, duration

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
        
def decode_webm_chunk(audio_bytes):
    process = subprocess.Popen(
        ['ffmpeg', '-i', 'pipe:0',
         '-f', 's16le', '-acodec', 'pcm_s16le',
         '-ac', '1', '-ar', '16000', 'pipe:1'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    pcm_data, _ = process.communicate(input=audio_bytes)
    audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)/32768.0
    return audio_np

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

    # trim silence: 
    vad = Vad(sample_rate=SAMPLING_RATE)

    waveform = vad(waveform)


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