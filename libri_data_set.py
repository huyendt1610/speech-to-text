# %%
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

import src.dataset as dataset
import warnings
import time

warnings.simplefilter('ignore')

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
    #print(tokenizer.vocab) # tokenizer nào cũng na ná nhau thôi, dùng cái này có sẵn đã có decoder trong đó rồi 

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

class MaskedConvd2d(nn.Conv2d): # inherit from nn.Conv2d
    def __init__(self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding=0, 
            bias=True, 
            **kwargs): 
        super(MaskedConvd2d, self).__init__(in_channels=in_channels, 
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            bias=bias,
                                            **kwargs)
    
    def forward(self, x, seq_lens): # define how data go through the model - the computing process of nn, seq_lens: valid lengths that come out of it 
        batch_size , channels, height, width = x.shape # standard form for a convolution (like for an image)
        output_seq_lens = self._compute_out_seq_len(seq_lens)

        conv_out = super().forward(x) # using the original forward method of the original convolution on this data

        mask = torch.zeros(batch_size, output_seq_lens.max(), device=x.device) # device list: CPU/GPU => x, mask need to be on the same device to avoid errors

        for i, length in enumerate(output_seq_lens): 
            mask[i, :length] = 1

        mask = mask.unsqueeze(1).unsqueeze(1)

        conv_out = conv_out * mask

        return conv_out, output_seq_lens

    def _compute_out_seq_len(self, seq_lens): 
        return torch.floor((seq_lens + (2 * self.padding[1]) - (self.kernel_size[1] -1) -1) // self.stride[1]) + 1

# ### Convolutional Feature Extractor 
# 
# Create a stack of two convolutions with BatchNorm2d and the HardTanh Activation function 
# 
# The output of the convolutions will give the shape (Batch x Channels X Mel_features x time). To give this to the future RNN, we need to reshape it to (Batch x time x Channels * Mel_features)
# 
# from (B, C, H, W) => (B, C, H*W) ~ (B, embedding, tokens)
# => need to this format: (B, tokens, embedding)
# 
class ConvolutionFeatureExtractor(nn.Module): 
    def __init__(self, in_channels = 1, out_channels = 32): # default from Nvidia implementation
        super(ConvolutionFeatureExtractor, self).__init__()

        self.conv1 = MaskedConvd2d(in_channels, out_channels, kernel_size=(11, 41), stride=(2,2), padding=(5, 20), bias=False) # co BatchNorm roi => no need bias
        self.bn1 = nn.BatchNorm2d(out_channels) # normalize the batch 

        self.conv2 = MaskedConvd2d(out_channels, out_channels, kernel_size=(11, 21), stride=(2,1), padding=(5, 10), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) # normalize the batch 

        self.output_feature_dim = 20 
        self.conv_output_features = self.output_feature_dim * out_channels

    def forward(self, x, seq_lens): 
        
        #print("Before conv1: ", x.shape)
        #print(seq_lens)

        x, seq_lens = self.conv1(x, seq_lens)
        x = self.bn1(x)
        x = torch.nn.functional.hardtanh(x)

       # print("\nAfter Convolution 1 layer: ", x.shape)
        #print(seq_lens)

        x, seq_lens = self.conv2(x, seq_lens)
        x = self.bn2(x)
        x = torch.nn.functional.hardtanh(x)

        #x = x.permute(0, 3, 1, 2) # change dimension orders from (0, 1, 2, 3) => (0, 3, 1, 2)
        #print("\nAfter Convolution 2 layer: ", x.shape)
        #print(seq_lens)

        x = x.permute(0, 3, 1, 2) # change dimension orders from (0, 1, 2, 3) => (0, 3, 1, 2)
        #print("\nAfter permute: ", x.shape)
        x = x.flatten(2) # flatten from the dimension with index = 2
        #print("\nAfter flatten: ", x.shape) 

        return x, seq_lens
    
class RNNLayer(nn.Module): 
    def __init__(self, input_size, 
                 hidden_size = 512): # what the invading implementaion used 
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.rnn = nn.LSTM( # Long Short-Term Memory
            input_size = input_size, # no of features at each timestep 
            hidden_size=hidden_size, # no of neuron at hidden state 
            batch_first=True, 
            bidirectional=True, # when training the rnn, future steps can look in the past steps and vice versa as the entire input is passed at once
        )

        self.layernorm = nn.LayerNorm(2 * hidden_size) # need *2 as for both forward and backward direction with bidirectional=True

    def forward(self, x, seq_lens):
        batch, seq_len, embed_dim = x.shape 

        # packing to be used in rnn to save time processing 
        packed_x = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True)

        out, _ = self.rnn(packed_x)

        # unpacking 
        x, _ = nn.utils.rnn.pad_packed_sequence(out, total_length=seq_len, batch_first=True )

        x = self.layernorm(x) # normalize data

        return x 

# ### Put it together 
# 
# Put it all together to create our DeepSpeech2 model 

class DeepSpeech2(nn.Module): 
    def __init__(self, 
                conv_in_channels = 1, 
                conv_out_channels = 32, 
                rnn_hidden_size = 512, # reduce model size
                rnn_depth = 5 # reduce model size
                ):
        
        super().__init__()

        self.feature_extractor = ConvolutionFeatureExtractor(conv_in_channels, conv_out_channels)

        self.output_hidden_features = self.feature_extractor.conv_output_features # 640 features 

        self.rnns = nn.ModuleList(  # a bunch of RNN 
            [
                # first layer: input_size = 640 
                # after that: input_size = 512 * 2 = 1024
                RNNLayer(input_size=self.output_hidden_features if i == 0 else 2 * rnn_hidden_size, hidden_size=rnn_hidden_size)
                for i in range(rnn_depth)
            ]
        )

        self.head = nn.Sequential( # output: probabilities for each vocab
            nn.Linear(2 * rnn_hidden_size, rnn_hidden_size), # input, output
            nn.Hardtanh(),
            nn.Linear(rnn_hidden_size, tokenizer.vocab_size) # fully connected layer to predict what the letter is said at this timestep?
        )


    def forward(self, x, seq_lens): 
        x, final_seq_lens = self.feature_extractor(x, seq_lens)

        for rnn in self.rnns: 
            x = rnn(x, final_seq_lens) # after convolution layer => final_seq_lens will never change

        x = self.head(x)
        
        return x, final_seq_lens



def main():
   

    DATASET_ROOT = "C:/Users/HuyenDT/Downloads/LibriSpeech"
    os.listdir(DATASET_ROOT)

    # file hiện tại
    CURRENT_FILE = Path.cwd().resolve()  #Path(__file__).resolve()
    # project root
    PROJECT_ROOT = CURRENT_FILE.parents[0]
    DATA_DIR = PROJECT_ROOT / "data"
    DATASET_CACHE = DATA_DIR / "dataset_cache"
    
    ### Training agurments 
    BATCH_SIZE = 32
    TRAINING_ITERATIONS = 10 # 50000 # how many iterations 
    EVAL_ITERATIONS = 5  # 2500 # How often want to evaluate a learning reate 
    LEARNING_RATE = 1e-4 # 10^(-4)
    NUM_WORKERS = 2 # no of CPU (if has data preload => set NUM_WORKERS = 0)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ### Data loaders ###
    trainset = dataset.LibrispeechDataset(path_to_data_root=DATASET_ROOT, include_splits=["train-clean-100"], is_from_cached = False, save_cached=False, is_augment=True)
    sampleset = dataset.LibrispeechDataset(path_to_data_root=DATASET_ROOT, include_splits=["dev-clean"], is_from_cached = False, save_cached=False)
    #trainset = dataset.LibrispeechDataset(path_to_data_root=DATASET_ROOT, include_splits=["train-clean-100"], is_from_cached = False, cache_version=1, cached_path=DATASET_CACHE)
    #sampleset = dataset.LibrispeechDataset(path_to_data_root=DATASET_ROOT, include_splits=["dev-clean"], is_from_cached = False, cache_version=2, cached_path=DATASET_CACHE)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fun, num_workers=NUM_WORKERS,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2)
    testloader = DataLoader(sampleset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fun, num_workers=NUM_WORKERS,  
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2) # , persistent_workers=True when set multiple workers

    ### Define the model ###
    model = DeepSpeech2(conv_in_channels=1, 
                        conv_out_channels=32,
                        rnn_hidden_size= 128,  #512
                        rnn_depth = 2 # 5 
                        ).to(DEVICE)

    # numel: number of elements
    params = sum([p.numel() for p in model.parameters()])
    print("Total training Parameter: ", params)

    ### Optimizer ###
    optimizer = optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=TRAINING_ITERATIONS)

    # model.load_state_dict(torch.load("best_weights_128-2-Wer_0.86.pt",weights_only=True)) # best_weights_512_3rnn, best_weights_128_2rnn

    best_val_loss = np.inf 
    train = True 
    completed_steps = 0 
    train_his_loss, validation_his_loss = [], []
    pbar = tqdm(range(TRAINING_ITERATIONS))
    start_total = time.time()


    while train: 
        training_loss = []
        validation_loss = []
        start_epoch = time.time()

        for batch in trainloader: 
            # print(batch)
            # input_lengths: seq_lens after conv layer: where is actual input, where is padding 
            logits, input_lengths = model(batch["input_values"].to(DEVICE), batch["seq_lens"])
            
            #print(logits.shape)
            #print(input_lengths)

            log_probs = nn.functional.log_softmax(logits, dim=-1) # dim=-1: apply softmax for the last dimension [batch_size, num_classes],
            #print(log_probs.shape) # => [batch_size, seq_lens, num_classes]
            log_probs = log_probs.transpose(0,1) # but CTC expect: [Time, batch_size, num_classes] => need to transpose two first dimensions

            # print(len(batch["labels"]), sum(batch["target_lengths"])) # the same number 

            loss = nn.functional.ctc_loss(
                log_probs=log_probs,
                targets=batch["labels"].to(DEVICE),
                input_lengths=input_lengths, 
                target_lengths=batch['target_lengths'].to(DEVICE),
                blank=tokenizer.pad_token_id, 
                reduction="mean"
            )

            # print(loss)

            loss.backward() # backpropagation
            optimizer.step() # update weights
            optimizer.zero_grad(set_to_none=True) # reset gradients of previous batch, which does not have effect to the next batch
            scheduler.step() # update learning rate 

            training_loss.append(loss.item())
            completed_steps += 1 
            pbar.update(1)

            if completed_steps % EVAL_ITERATIONS == 0: 
                print(f"Evaluating...{time.time()}")
                model.eval() 

                for batch in tqdm(testloader): 
                    ### Pass through model and get input_lengths (pocst convolutions) and logits ###
                    with torch.no_grad(): # tells PyTorch to not calculate the gradients in this block 
                        logits, input_lengths = model(x=batch["input_values"].to(DEVICE), seq_lens=batch["seq_lens"])

                        #CTC expects log probabilities 
                        log_probs = nn.functional.log_softmax(logits, dim=-1)

                        # CTC also expects (TxBxC), we have (BxTxC)
                        log_probs = log_probs.transpose(0,1)

                        # Compute CTC loss 
                        loss = nn.functional.ctc_loss(
                            log_probs=log_probs,
                            targets=batch["labels"].to(DEVICE),
                            input_lengths=input_lengths, 
                            target_lengths=batch['target_lengths'].to(DEVICE),
                            blank=tokenizer.pad_token_id, 
                            reduction="mean"
                        )

                        # Store Loss
                        validation_loss.append(loss.item())
                
                training_loss_mean = np.mean(training_loss)
                valid_loss_mean = np.mean(validation_loss)

                # log to hist
                train_his_loss.append(training_loss_mean)
                validation_his_loss.append(valid_loss_mean)

                # Save model if val loss decreases 
                if valid_loss_mean < best_val_loss:
                    print(f"---Saving model---{time.time()}")
                    torch.save(model.state_dict(), "best_weights.pt")
                    torch.save({
                        'epoch': completed_steps,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, "checkpoint.pt")
                                    
                    best_val_loss = valid_loss_mean

                print("Training Loss:", training_loss_mean)
                print("Validation Loss:", valid_loss_mean)

                # Rest list 
                training_loss = []
                validation_loss = []

                # Set Model to Training mode
                model.train() 

            if completed_steps >= TRAINING_ITERATIONS: 
                train = False 
                print("Completed!", f"Total training time: {time.time() - start_total:.2f}s")
                break
    
if __name__ == "__main__":
    main()