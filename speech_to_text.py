# pip3 install speechrecognition 
# pip3 install pyttsx3 // Text To Speech x3 //pip install pyaudio

import speech_recognition as sr 
import pyttsx3 # pip 

r = sr.Recognizer() # Initialize the recognizer

def record_text():
    # Loop in case of errors
    while(1): 
        try: 
            # use the mic as source for input
            with sr.Microphone() as mic:
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
    return 

def output_text(text):
    with open("output.txt", "a") as f:
        f.write(text)
        f.write("\n")
    return 

while (1):
    text = record_text() 
    output_text(text) 

    print("wrote text")