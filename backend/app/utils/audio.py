
import torch
import torchaudio
from torchaudio.transforms import Vad
import subprocess
import numpy as np 
from app.config import settings
import io 

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

    if orig_sr != settings.SAMPLING_RATE:
        #print(f"Changed the sampling rate from {orig_sr}: {SAMPLING_RATE}")
        waveform = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=settings.SAMPLING_RATE) # re-sample to 16.000

    # trim silence: reduce audio length, increase inference speed, improve accuracy 
    # trigger_level: level to determind speech; default 7, but in [6, 10], if too much noise => incrase 8, else if miss speech => reduce to 6
    # VAD = Voice Activity Detection
    vad = Vad(sample_rate=settings.SAMPLING_RATE) 
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
