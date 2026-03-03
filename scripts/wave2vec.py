import torch
import numpy as np
import sounddevice as sd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load mô hình Wav2Vec2 (pre-trained cho tiếng Anh)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Hàm ghi âm từ microphone
def record_audio(duration=5, fs=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # chờ ghi xong
    return audio.flatten()

# Hàm chuyển audio sang text
def speech_to_text(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits # ** from dictionary to keyword arguments, in case too many args => no need to pass one-by-one
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

if __name__ == "__main__":
    # Ghi âm 5 giây và chuyển sang text
    audio = record_audio(duration=5)
    text = speech_to_text(audio)
    print("Transcription:", text)