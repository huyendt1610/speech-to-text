import torch
import torchaudio
import sounddevice as sd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from src.utils import get_libri_file_list
import io 

# Load model Wav2Vec2 (pre-trained for English)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# record_audio
def record_audio(duration=5, fs=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  #
    return audio.flatten()

# speech_to_text
def speech_to_text(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits # ** from dictionary to keyword arguments, in case too many args => no need to pass one-by-one
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

if __name__ == "__main__":
    # Record 5 sec and convert to text
    # audio = record_audio(duration=5)
    # text = speech_to_text(audio)
    # print("Transcription:", text)

    filelist = get_libri_file_list()
    file_idx = 0 

    audio_path, org_transcript = filelist[file_idx] 
    # print(audio_path)
    # print(org_transcript)
    audio, sr = torchaudio.load(audio_path, normalize=True)
    text = speech_to_text(audio.flatten())
    print("Transcription:", text)