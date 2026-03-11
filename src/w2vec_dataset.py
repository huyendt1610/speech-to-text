import os 
import numpy as np 
import pandas as pd 
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor 
from dataclasses import dataclass

from .w2vec_utils import (
    compute_span_mask, 
    sample_negative_indices, 
    compute_sub_attention_mask,
    Wave2Vec2Config
)

class W2V2LibriDataset(Dataset):
    def __init__(self, 
                 path_to_data_root, 
                 include_splits = ["dev-clean"], # ["train-clean-100", "train-clean-360", "train-other-500"], 
                 max_audio_duration=20.0, 
                 min_audio_duration=2.0,
                 sampling_rate = 16000, 
                 num_audio_channels = 1,
                 truncate_audio=True, 
                 return_transcripts=True,
                 hf_model_name="facebook/wav2vec2-base"):
        
        if isinstance(include_splits, str): # if it is a string 
                include_splits = [include_splits] # to make sure include_splits is a list, even if it is a string => 1 item
            
        self.sampling_rate = sampling_rate 
        self.return_transcripts = return_transcripts
        self.truncate_audio = truncate_audio
        self.num_audio_channels = num_audio_channels
        self.min_audio_samples = int(min_audio_duration * sampling_rate)
        self.max_audio_samples = int(max_audio_duration * sampling_rate)

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
                    audio_durations = pd.read_csv(os.path.join(path_to_section, "audio_durations.csv"))
                    audio_durations_dict = audio_durations.set_index("root")["duration"].to_dict()

                    with open(os.path.join(path_to_section, transcript_file), "r") as f: 
                        transcripts = f.readlines()

                    for line in transcripts: 
                        split_line = line.split() # default is space => return an array
                        audio_root = split_line[0]
                        audio_file = audio_root + ".flac"
                        full_path_to_audio_file = os.path.join(path_to_section, audio_file)
                        transcript = " ".join(split_line[1:]).strip()
                        
                        duration = audio_durations_dict[audio_root]
                        
                        if (duration >= min_audio_duration) and (duration <= max_audio_duration or truncate_audio):
                            self.librispeech_data.append(
                                (full_path_to_audio_file, transcript)
                            )
                        

        if return_transcripts: 
            self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(hf_model_name)
            
            


    def __len__(self):
        return len(self.librispeech_data)

    def __getitem__(self, index):
        path_to_audio, transcript = self.librispeech_data[index]
        audio, org_sr = torchaudio.load(path_to_audio)

        if self.truncate_audio: 
            audio = audio[:, :self.max_audio_samples] # [channels, num_samples]

        if org_sr != self.sampling_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=org_sr, new_freq=self.sampling_rate)

        if self.num_audio_channels == 1: 
            audio = audio.squeeze() # remove the dimensions that have values = 1

        normalized_audio = (audio - audio.mean())/(np.sqrt(audio.var() + 1e-7)) # nomarlize 

        if self.return_transcripts: 
            tokened_transcripts = torch.tensor(self.tokenizer.encode(transcript))
            sample = {
                "input_values": normalized_audio, 
                "labels": tokened_transcripts
                }
        else:
            sample = {
                "input_values": normalized_audio
                }
        return sample 



if __name__ == "__main__":
    path_to_data = "C:/Users/HuyenDT/Downloads/LibriSpeech"
    dataset = W2V2LibriDataset(path_to_data, return_transcripts=False)
    sample = next(iter(dataset))
    print(sample)