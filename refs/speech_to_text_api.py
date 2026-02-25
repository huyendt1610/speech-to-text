# pip3 install speechrecognition 
# pip3 install pyttsx3 // Text To Speech x3 
# //pip install pyaudio

import speech_recognition as sr 
import pyttsx3 # pip 
from pydub import AudioSegment # pip install pydub
from pydub.utils import make_chunks
import os 

r = sr.Recognizer() # Initialize the recognizer

def record_text():
    # Loop in case of errors
    try: 
        # use the mic as source for input
        with sr.Microphone() as mic:
            print("Listening...")
            # prepare recognizer to receive input 
            r.adjust_for_ambient_noise(mic, duration=0.2) 
        
            # listen for the user's input 
            au = r.listen(mic)

            # using google to recognize audio 
            txt = r.recognize_google(au)

            return txt 

    except sr.RequestError as e:
        print(f"Could not request results: {e}")
    except sr.UnknownValueError: 
        print("Unknown error occurred") 

def output_text(text):
    with open("output.txt", "a") as f:
        f.write(text)
        f.write("\n")
    return 

def decode_audio(path):
    with sr.AudioFile(path) as source:
        audio = r.record(source)   # đọc toàn bộ file

    text = r.recognize_google(audio, language="en-US")
    print(text)

def decode_whole_file(path, chunk_length_ms=30_000, language="en-US", temp_folder="temp_chunks"):
    os.makedirs(temp_folder, exist_ok=True)

    # Load file lớn
    audio = AudioSegment.from_file(path)

    # Convert to mono 16kHz (Google recommends)
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Chunk size (ms), ví dụ 30 giây = 30*1000 ms
    #chunk_length_ms = 30 * 1000
    chunks = make_chunks(audio, chunk_length_ms)

    r = sr.Recognizer()
    full_text = ""
    
    # Lưu tạm chunks ra file
    for i, chunk in enumerate(chunks):
        chunk_name = f"temp/chunk_{i}.wav"
        chunk.export(chunk_name, format="wav")
    
    for i, chunk in enumerate(chunks):
        chunk_name = os.path.join(temp_folder, f"chunk_{i}.wav")
        chunk.export(chunk_name, format="wav")
        
        with sr.AudioFile(chunk_name) as source:
            audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data, language=language)
            full_text += text + " "
        except sr.UnknownValueError:
            print(f"[Warning] Chunk {i} could not be understood")
        except sr.RequestError as e:
            print(f"[Error] Could not request results from Google; {e}")
    
    # Optional: clean up temp folder
    # import shutil
    # shutil.rmtree(temp_folder)
    
    return full_text.strip()

if __name__ == "__main__":
    while (1):
        text = record_text() 
        # output_text(text) 

        print("Recognized text: ", text)