import os 
import torch
import torchaudio
import pandas as pd
import numpy as np 
import shutil
from transformers import Wav2Vec2CTCTokenizer 
import torchaudio.transforms as T 


def get_libri_file_list(path_to_data_root = "C:/Users/HuyenDT/Downloads/LibriSpeech", 
                        include_splits = ["dev-clean"]):
    librispeech_data = []
    for s in include_splits: 
        path_to_split = os.path.join(path_to_data_root, s)  # format: speaker/section/audio
        
        for speaker in os.listdir(path_to_split): 
            path_to_speaker = os.path.join(path_to_split, speaker)
            # print(speaker)

            for section in os.listdir(path_to_speaker): 
                path_to_section = os.path.join(path_to_speaker, section)
                files = os.listdir(path_to_section)

                transcript_file = [path for path in files if ".txt" in path][0]
                with open(os.path.join(path_to_section, transcript_file), "r") as f: 
                    transcripts = f.readlines()

                for line in transcripts: 
                    split_line = line.split() # default is space => return an array
                    audio_root = split_line[0]
                    audio_file = audio_root + ".flac"
                    full_path_to_audio_file = os.path.join(path_to_section, audio_file)
                    transcript = " ".join(split_line[1:]).strip()

                    librispeech_data.append(
                        (full_path_to_audio_file, transcript)
                    )
    
    return librispeech_data

def calculate_audio_durations(path_to_data_root = "C:/Users/HuyenDT/Downloads/LibriSpeech", 
                        include_splits = ["dev-clean"]):
    
    if include_splits == "": 
        include_splits = [entry.name for entry in os.scandir(path_to_data_root) if entry.is_dir()]

    for s in include_splits: 
        path_to_split = os.path.join(path_to_data_root, s)  # format: speaker/section/audio
        
        for speaker in os.listdir(path_to_split): 
            path_to_speaker = os.path.join(path_to_split, speaker)
            # print(speaker)

            for section in os.listdir(path_to_speaker): 
                path_to_section = os.path.join(path_to_speaker, section)
                files = os.listdir(path_to_section)

                audio_durations = {
                    "root": [],
                    "duration": []
                }
                audio_durations_path = os.path.join(path_to_section, "audio_durations.csv")

                transcript_file = [path for path in files if ".txt" in path][0]
                with open(os.path.join(path_to_section, transcript_file), "r") as f: 
                    transcripts = f.readlines()
                for line in transcripts: 
                    split_line = line.split() # default is space => return an array
                    audio_root = split_line[0]
                    audio_file = audio_root + ".flac"
                    full_path_to_audio_file = os.path.join(path_to_section, audio_file)
                    waveform, sample_rate = torchaudio.load(full_path_to_audio_file)
                    duration = waveform.shape[1] / sample_rate 
                    audio_durations["root"].append(audio_root)
                    audio_durations["duration"].append(duration)

                
                df = pd.DataFrame(audio_durations)
                df.to_csv(audio_durations_path, index=False, header=["root", "duration"])
                print(f"Path: {path_to_section}... Done")


def get_max_cached_version(cached_path, cache_version): 
    if(cache_version > 0): 
        return cache_version
    else: 
        return np.max([int(i) for i in os.listdir(cached_path)]) 
    
def build_cache(path_to_data_root = "C:/Users/HuyenDT/Downloads/LibriSpeech", 
                include_splits = ["dev-clean"], 
                sampling_rate = 1600,
                cached_path = "dataset_cache", 
                cache_version = 0):
    # check cached version 
    max_version = get_max_cached_version(cached_path, cache_version)
    path_to_cache_version = os.path.join(cached_path, str(max_version))
    if os.path.exists(path_to_cache_version):
        shutil.rmtree(path_to_cache_version)
    os.makedirs(path_to_cache_version, exist_ok=True)

    librispeech_data = get_libri_file_list(path_to_data_root, include_splits)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
    SAVE_BATCH = 2000
    audio2mels = T.MelSpectrogram( # default n_fft & hanning_window
                sample_rate = sampling_rate, # tấn suất lấy mẫu của audio: esim: 16000 Hz = 1 giây có 16000 mẫu
                n_mels=80 # Mel filter banks: (optimal value) Output sẽ có 80 hàng (80 tần số mel), => chia trục tần số thành 80 dải Mel.
                        # 40: lost data, 128-256: heavy, more RAM, and training time,
                        # range map from Hz to Mel 
            )

    amp2db = T.AmplitudeToDB( # covert from linear-scale mel spectrogram to log-mel
        top_db=80.0
    )

    #print(max_version)
    # data = []
    buffer = []
    file_id = 0
    for path, tran in librispeech_data: 
        #print(i, path, tran)
        audio, orig_sr = torchaudio.load(path, normalize=True)
        if orig_sr != sampling_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=sampling_rate) # re-sample to 16.000

        mel = audio2mels(audio) # to MelSpectrogram
        mel = amp2db(mel) 
        mel = (mel - mel.mean())/(mel.std() + 1e-6) # 1e-6 to avoid deviding by zero, to nomalize

        tokenized_transcript = torch.tensor(tokenizer.encode(tran))
        item = {
                "input_values": mel[0].T, # to feed time dimension first, then feature dimension after
                "labels": tokenized_transcript
            }
        # data.append(item)
        buffer.append(item)

        if len(buffer) == SAVE_BATCH:
                # save to cache
                path_to_save = os.path.join(path_to_cache_version, f"data_{file_id}.pt") # audioname.pt
                torch.save(buffer,path_to_save)

                # torch.save(buffer, f"data_{file_id}.pt")
                buffer = []
                file_id += 1
                
    # save remaining parts
    if buffer:
        path_to_save = os.path.join(path_to_cache_version, f"data_{file_id}.pt") # audioname.pt

        torch.save(buffer,path_to_save)
    
    # save self.librispeech_data 
    path_to_save = os.path.join(path_to_cache_version, "speech.pt") 
    torch.save(librispeech_data, path_to_save)

if __name__ == "__main__":
    # l = get_libri_file_list()
    # print(len(l))
    calculate_audio_durations(include_splits="")

    

