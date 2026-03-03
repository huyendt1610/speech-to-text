import random
import datasets 
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing
from pathlib import Path
#import sounddevice as sd 

def collate_fn(batch): 
    # Get max audio length 
    max_audio_len = max([item["audio"]])