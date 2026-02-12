import speech_recognition 
import pyttsx3 

recognizer = speech_recognition.Recognizer() 

while True: 
    try: 
        with speech_recognition.Microphone() as mic: 

            # adjust_for_ambient_noise = "tự động cân bằng ngưỡng nhận dạng giọng nói để bỏ qua tiếng ồn nền"
            # 1. listen to measure the noise 
            # 2. set energy_threshold: the threshold that mic consider that is "the voice"
            recognizer.adjust_for_ambient_noise(mic, duration = 0.2)
            print(recognizer.energy_threshold)

            print('Listening....')

            # When users stop talking => 
            audio = recognizer.listen(mic) 

            # Dùng API của Google Speech Recognition để phân tích âm thanh trong audio.
            text = recognizer.recognize_google(audio) # Ctrl+Shift+P -> de cho kieu kernel, intepreter

            text = text.lower() 

            print(f"Result: {text}")
    except speech_recognition.UnknownValueError: 
        recognizer = speech_recognition.Recognizer() 
        continue
    except speech_recognition.RequestError:
        print("Erro when connecting to API")
 