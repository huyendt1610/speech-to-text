# pip install torch torchaudio torchcodec 
# pip install transformers jiwer ipywidgets
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm # for displaying process bar
import torch 
import torch.nn as nn 
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T 
import torchaudio.functional as F 
from transformers import Wav2Vec2CTCTokenizer, get_cosine_schedule_with_warmup 
from jiwer import wer # to evaluate the model 
from pathlib import Path
import re
import shutil


tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
tokenizer # tokenizer nào cũng na ná nhau thôi, dùng cái này có sẵn đã có decoder trong đó rồi 

SAVE_BATCH = 2000

class LibrispeechDataset(Dataset):

    def __init__(self, 
                 path_to_data_root, 
                 include_splits = ["dev-clean"], # ["train-clean-100", "train-clean-360", "train-other-500"], 
                 sampling_rate = 16000, 
                 num_audio_channels = 1,
                 is_from_cached=False,
                 cached_path = "dataset_cache",
                 cache_version = 0): 

        self.sampling_rate = sampling_rate 
        self.num_audio_channels = num_audio_channels

        # preload data 
        self.data = []

        if (is_from_cached and cache_version > 0 and os.path.exists(os.path.join(cached_path, str(cache_version)))): # Load from cache 
            print("Loading cached dataset...")
            folder_path = os.path.join(cached_path, str(cache_version))
            # self.data = [torch.load(os.path.join(folder_path, f)) for f in os.listdir(folder_path)]
            pattern = re.compile(r"^data_.*")
           
            for f in os.listdir(folder_path):
                if pattern.match(f): 
                    self.data.extend(torch.load(os.path.join(folder_path, f)))

            self.librispeech_data = torch.load(os.path.join(folder_path, "speech.pt"))

        else: 
            self.build_data_from_files(path_to_data_root, include_splits, cached_path, cache_version)

    def build_data_from_files(self, path_to_data_root, include_splits, cached_path, cache_version): 
            # Load from folders 
            if isinstance(include_splits, str): # if it is a string 
                include_splits = [include_splits] # to make sure include_splits is a list, even if it is a string => 1 item
            
            self.librispeech_data = []
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

                            self.librispeech_data.append(
                                (full_path_to_audio_file, transcript)
                            )


            # print(len(self.librispeech_data))
            # Create a transform to transfrom the audio waveform → Mel Spectrogram: display Frequency by time (Time × Frequency)
            # Waveform (1D signal) => STFT (Fourier Transform) -> Mel scaling -> Mel Spectrogram (2D tensor)
            # Mel scale: based on how people can hear the voice to separate into level (is log(Hz))
            # n_fft?, window_size? window_fn: torch.hann_window
            self.audio2mels = T.MelSpectrogram( # default n_fft & hanning_window
                sample_rate = self.sampling_rate, # tấn suất lấy mẫu của audio: esim: 16000 Hz = 1 giây có 16000 mẫu
                n_mels=80 # Mel filter banks: (optimal value) Output sẽ có 80 hàng (80 tần số mel), => chia trục tần số thành 80 dải Mel.
                        # 40: lost data, 128-256: heavy, more RAM, and training time,
            )

            self.amp2db = T.AmplitudeToDB(
                top_db=80.0
            )

            # check cached version 
            max_version = self.get_max_cached_version(cached_path, cache_version)
            path_to_cache_version = os.path.join(cached_path, str(max_version))
            if os.path.exists(path_to_cache_version):
                shutil.rmtree(path_to_cache_version)
            os.makedirs(path_to_cache_version, exist_ok=True)

            #print(max_version)
            buffer = []
            file_id = 0
            for path, tran in self.librispeech_data: 
                #print(i, path, tran)
                audio, orig_sr = torchaudio.load(path, normalize=True)
                if orig_sr != self.sampling_rate:
                    audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=self.sampling_rate) # re-sample to 16.000

                mel = self.audio2mels(audio) # to MelSpectrogram
                mel = self.amp2db(mel) 
                mel = (mel - mel.mean())/(mel.std() + 1e-6) # 1e-6 to avoid deviding by zero, to nomalize

                tokenized_transcript = torch.tensor(tokenizer.encode(tran))
                item = {
                        "input_values": mel[0].T, # to feed time dimension first, then feature dimension after
                        "labels": tokenized_transcript
                    }
                self.data.append(item)
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
            path_to_save = os.path.join(path_to_cache_version, f"speech.pt") 
            torch.save(self.librispeech_data, path_to_save)

    def get_max_cached_version(self, cached_path, cache_version): 
        if(cache_version > 0): 
            return cache_version
        else: 
            return np.max([int(i) for i in os.listdir(cached_path)]) 
    
    def __len__(self): 
        '''
            - DataLoader khow how many samples dataset has
            - To calculate the number of batch
            - Shuffle correctly
        '''
        return len(self.librispeech_data)
    
    def __getitem__(self, index):
        # path_to_audio, transcript = self.librispeech_data[index]
        # audio, orig_sr = torchaudio.load(path_to_audio, normalize=True) #normalize: true, convert into [-1, 1], normalize: false, audio bit is between [0, 255]
        # if orig_sr != self.sampling_rate:
        #     audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=self.sampling_rate) # re-sample to 16.000

        # # audio: waveform
        # mel = self.audio2mels(audio) # to MelSpectrogram
        # # print(audio.shape) => [1, 170400]
        # # print(mel.shape) => [1, 80, 853]: 80 bins, 
        # # 853 time steps: after sliding window of hann_window (400 windowsize with 200 overlap default value)

        # mel = self.amp2db(mel) # to amplitude to Decibel => see diff frequencies lighting up at the diff time steps
        
        # # plt.figure(figsize=(15,5))
        # # plt.imshow(mel[0])
        # # plt.show()

        # mel = (mel - mel.mean())/(mel.std() + 1e-6) # 1e-6 to avoid deviding by zero, to nomalize

        # tokenized_transcript = torch.tensor(tokenizer.encode(transcript))
        # # print(transcript)
        # # print(tokenized_transcript)

        # sample = {
        #     "input_values": mel[0].T, # to feed time dimension first, then feature dimension after
        #     "labels": tokenized_transcript
        #     # "audio": audio,
        #     #"transcript": transcript
        # }
        #  return sample

        return self.data[index]
       