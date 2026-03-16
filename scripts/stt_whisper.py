# pip install git+https://github.com/openai/whisper.git
# pip install torchaudio

import whisper # OpenAI
from pathlib import Path
from jiwer import wer
import sounddevice as sd
from src.utils import get_libri_file_list

import warnings
warnings.simplefilter('ignore')

    
def testLibriData(file_idx, la = "en"):
    # FILE_INDEX = 0 
    audio_path, org_transcript = filelist[file_idx]
    audio_path, org_transcript 

    audio_file = Path(audio_path)
    result = model.transcribe(str(audio_file), language=la)
    
    pred_transcript = str(result["text"]).upper()

    print("Transcribed text:", pred_transcript)
    print("Original text:", org_transcript)
    print("WER:", wer(org_transcript, pred_transcript))

def record_audio(duration=5, fs=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # chờ ghi xong
    return audio.flatten()

def speech_to_text(audio, la = "en"): 
    result = model.transcribe(audio, language=la)
    pred_transcript = str(result["text"]).upper()

    print("Transcribed text:", pred_transcript)


if __name__ == "__main__":

    model = whisper.load_model("tiny")
    filelist = get_libri_file_list()
    valid_languages = ["en", "es", "fr", "de", "it", "pt", "nl", "ru", "ar", "zh", "ja", "ko", "vi", "fi", "sv", "da", "no", "tr", "pl", "ro", "hu", "el", "cs", "sk", "he", "th", "uk", "hi", "bn", "ms", "id"] 
    language = "en"
    c = True

    while c: 
        print("-"+ language + "-"*30)
        print("1. Get from testset")
        print("2. Record an audio")
        print("3. Config the language")
        print("4. Exit")
        
        rs = input("Choose a program: ")
        
        if not rs.isdigit():
            print("Invalid input, please enter a number")
            continue
        
        rs = int(rs)

        if rs == 1:
            idx = int(input("-- Choose an index: "))
            testLibriData(idx, language)
        elif rs == 2:
            audio = record_audio(duration=5)
            text = speech_to_text(audio, language)
        elif rs == 3:
            selected_language = input("-- Choose a language: en, fi, vi...: ")
            if selected_language not in valid_languages:
                print("Invalid input, please enter again")
                continue
            language = selected_language
        elif rs == 4:
            c = False
        else:
            print("Invalid choice")

