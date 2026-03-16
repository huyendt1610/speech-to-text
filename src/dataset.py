# pip install torch torchaudio torchcodec 
# pip install transformers jiwer ipywidgets
import os 
import numpy as np 
import torch 
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T 
from transformers import Wav2Vec2CTCTokenizer 
import re
import shutil
import random

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")

class LibrispeechDataset(Dataset):

    def __init__(self, 
                 path_to_data_root, 
                 include_splits = ["dev-clean"], # ["train-clean-100", "train-clean-360", "train-other-500"], 
                 is_augment = False,
                 sampling_rate = 16000, 
                 num_audio_channels = 1,
                 is_from_cached=False,
                 cached_path = "dataset_cache",
                 cache_version = 0): 

        self.sampling_rate = sampling_rate 
        self.num_audio_channels = num_audio_channels
        self.is_augment = is_augment

        # preload data 
        self.data = []

        # Load from cache 
        if (is_from_cached and cache_version > 0 and os.path.exists(os.path.join(cached_path, str(cache_version)))): 
            print("Loading cached dataset...")
            folder_path = os.path.join(cached_path, str(cache_version))
            # self.data = [torch.load(os.path.join(folder_path, f)) for f in os.listdir(folder_path)]
            pattern = re.compile(r"^data_.*")
           
            for f in os.listdir(folder_path):
                if pattern.match(f): 
                    self.data.extend(torch.load(os.path.join(folder_path, f)))

            self.librispeech_data = torch.load(os.path.join(folder_path, "speech.pt"))

        else: 
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
                        # range map from Hz to Mel 
            )

            self.amp2db = T.AmplitudeToDB( # covert from linear-scale mel spectrogram to log-mel
                top_db=80.0
            )

            self.audio_aug = AudioAugment()
            self.spec_aug  = SpecAugment()
            
    def __len__(self): 
        '''
            - DataLoader khow how many samples dataset has
            - To calculate the number of batch
            - Shuffle correctly
        '''
        return len(self.librispeech_data)
    
    def __getitem__(self, index):
        path_to_audio, transcript = self.librispeech_data[index]
        audio, orig_sr = torchaudio.load(path_to_audio, normalize=True) #normalize: true, convert into [-1, 1], normalize: false, audio bit is between [0, 255]
        if orig_sr != self.sampling_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=self.sampling_rate) # re-sample to 16.000

        if (self.is_augment): # only apply augmentation for training data
            audio = self.audio_aug(audio)

        # audio: waveform
        mel = self.audio2mels(audio) # to MelSpectrogram
        # print(audio.shape) => [1, 170400]
        # print(mel.shape) => [1, 80, 853]: 80 bins, 
        # 853 time steps: after sliding window of hann_window (400 windowsize with 200 overlap default value)

        mel = self.amp2db(mel) # to amplitude to Decibel => see diff frequencies lighting up at the diff time steps
        
        # plt.figure(figsize=(15,5))
        # plt.imshow(mel[0])
        # plt.show()

        if (self.is_augment): # only apply augmentation for training data
            mel = self.spec_aug(mel)

        mel = (mel - mel.mean())/(mel.std() + 1e-6) # 1e-6 to avoid deviding by zero, to nomalize

        tokenized_transcript = torch.tensor(tokenizer.encode(transcript))
        # print(transcript)
        # print(tokenized_transcript)

        sample = {
            "input_values": mel[0].T, # to feed time dimension first, then feature dimension after
            "labels": tokenized_transcript
            # "audio": audio,
            #"transcript": transcript
        }

        return sample

        # return self.data[index]

def collate_fun(batch): # combine multiple samples into batchs
    #print(batch[0]["input_values"].shape)  # time steps of each audio is 853
    #print(batch[1]["input_values"].shape)  # each has diff length 

    batch = sorted(batch, key = lambda x: x["input_values"].shape[0], reverse=True)
    # print(batch[0]["input_values"].shape)  # after sort 
    # print(batch[1]["input_values"].shape) 
    

    batch_mels = [sample["input_values"] for sample in batch]
    batch_transcripts = [sample["labels"] for sample in batch]

    seq_lens = torch.tensor([b.shape[0] for b in batch_mels], dtype=torch.long)

    spectrograms = torch.nn.utils.rnn.pad_sequence(batch_mels, batch_first=True, padding_value=0)
    # print(spectrograms.shape) # [5, 853, 80]: batch_size, seq length, hidden size 
    
    spectrograms = spectrograms.unsqueeze(1) # add 1 more dimension after the first dimension, unsqueeze(0): at the beginning, unsqueeze(-1): at the end 
    # print(spectrograms.shape) # [5, 1, 853, 80]

    spectrograms = spectrograms.transpose(-1, -2) # switch the position of 2 last dimensions

    target_lengths = torch.tensor([len(t) for t in batch_transcripts], dtype=torch.long)
    packed_transcripts = torch.cat(batch_transcripts)

    # print(target_lengths)
    # print(packed_transcripts)
    batch = {
        "input_values": spectrograms,
        "seq_lens": seq_lens, 
        "labels": packed_transcripts, 
        "target_lengths": target_lengths
    }

    return batch
  
class SpecAugment:
    def __init__(self):
        self.freq_mask = T.FrequencyMasking(15)
        self.time_mask = T.TimeMasking(35)

    def __call__(self, spec):
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        return spec



class AudioAugment:
    def __init__(
        self,
        sample_rate=16000,
        speed_prob=0.5,
        noise_prob=0.3,
        gain_prob=0.3,
        shift_prob=0.3,
    ):
        self.sample_rate = sample_rate
        self.speed_prob = speed_prob
        self.noise_prob = noise_prob
        self.gain_prob = gain_prob
        self.shift_prob = shift_prob

    # ---------- SPEED ----------
    def speed_perturb(self, waveform):
        speed = random.choice([0.9, 1.0, 1.1])

        if speed == 1.0:
            return waveform

        resampler = torchaudio.transforms.Resample(
            self.sample_rate,
            int(self.sample_rate * speed)
        )

        waveform = resampler(waveform)

        resampler_back = torchaudio.transforms.Resample(
            int(self.sample_rate * speed),
            self.sample_rate
        )

        return resampler_back(waveform)
    
    # ---------- NOISE ----------
    def add_noise(self, waveform):
        noise = torch.randn_like(waveform)
        noise_level = random.uniform(0.001, 0.01)
        return waveform + noise * noise_level

    # ---------- GAIN ----------
    def random_gain(self, waveform):
        gain = random.uniform(0.7, 1.3)
        return waveform * gain

    # ---------- TIME SHIFT ----------
    def time_shift(self, waveform):
        shift = int(random.uniform(-0.1, 0.1) * self.sample_rate)
        return torch.roll(waveform, shifts=shift, dims=1)

    # ---------- APPLY ----------
    def __call__(self, waveform):

        if random.random() < self.speed_prob:
            waveform = self.speed_perturb(waveform)

        if random.random() < self.noise_prob:
            waveform = self.add_noise(waveform)

        if random.random() < self.gain_prob:
            waveform = self.random_gain(waveform)

        if random.random() < self.shift_prob:
            waveform = self.time_shift(waveform)

        return waveform
    