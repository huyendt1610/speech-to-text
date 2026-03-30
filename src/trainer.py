# pip install torch torchaudio torchcodec 
# pip install transformers jiwer ipywidgets
import os 
import numpy as np 
from tqdm import tqdm # for displaying process bar
import torch 
import torch.nn as nn 
from torch import optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2CTCTokenizer, get_cosine_schedule_with_warmup 
from pathlib import Path

import src.dataset as dataset
from src.dataset import collate_fun
from src.model import DeepSpeech2
import warnings
import time
# import importlib

# importlib.reload(dataset)

warnings.simplefilter('ignore')


def main():

    # file hiện tại
    CURRENT_FILE = Path.cwd().resolve()  #Path(__file__).resolve()
    # project root
    PROJECT_ROOT = CURRENT_FILE.parents[0]
    DATA_DIR = PROJECT_ROOT / "data"
    DATASET_CACHE = DATA_DIR / "dataset_cache"

    DATASET_ROOT = "./data/LibriSpeech"
    os.listdir(DATASET_ROOT)
    
    ### Training agurments 
    BATCH_SIZE = 32
    TRAINING_ITERATIONS = 20000 # 50000 # how many iterations 
    EVAL_ITERATIONS = TRAINING_ITERATIONS//10  # 2500 # How often want to evaluate a learning reate 
    LEARNING_RATE = 1e-4 # 10^(-4)
    NUM_WORKERS = 6 # no of CPU (if has data preload => set NUM_WORKERS = 0)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WARMUPS_STEPS = np.floor(0.01 * TRAINING_ITERATIONS) 
    # Phase 1: Warmup, get 1% of trainining steps to increase LEARNING_RATE from 0 to LEARNING_RATE, 0.01 * 20000 = 200 steps
    # Phase 2: Cosine decay, then decrease learning rate by Cosine function, to smooth learning rate 

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")

    ### Data loaders ###
    trainset = dataset.LibrispeechDataset(path_to_data_root=DATASET_ROOT, include_splits=["train-clean-100", "train-clean-360"], is_from_cached = False, is_augment=True)
    sampleset = dataset.LibrispeechDataset(path_to_data_root=DATASET_ROOT, include_splits=["dev-clean"], is_from_cached = False)
    #trainset = dataset.LibrispeechDataset(path_to_data_root=DATASET_ROOT, include_splits=["train-clean-100"], is_from_cached = False, cache_version=1, cached_path=DATASET_CACHE)
    #sampleset = dataset.LibrispeechDataset(path_to_data_root=DATASET_ROOT, include_splits=["dev-clean"], is_from_cached = False, cache_version=2, cached_path=DATASET_CACHE)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, 
                            shuffle=True, 
                            collate_fn=collate_fun, 
                            num_workers=NUM_WORKERS,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2)
    testloader = DataLoader(sampleset, batch_size=BATCH_SIZE, 
                            shuffle=True, collate_fn=collate_fun, 
                            num_workers=NUM_WORKERS,  
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2) # , persistent_workers=True when set multiple workers

    ### Define the model ###
    model = DeepSpeech2(conv_in_channels=1, 
                        conv_out_channels=32,
                        rnn_hidden_size= 128,  #512
                        rnn_depth = 2, # 5 
                        tokenizer=tokenizer
                        ).to(DEVICE)

    # numel: number of elements
    params = sum([p.numel() for p in model.parameters()])
    print("Total training Parameter: ", params)

    ### Optimizer ###
    optimizer = optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=NUM_WARMUPS_STEPS, num_training_steps=TRAINING_ITERATIONS)

    # model.load_state_dict(torch.load("best_weights.pt",weights_only=True)) # best_weights_512_3rnn, best_weights_128_2rnn
    checkpoint = torch.load("checkpoint.pt", weights_only =False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint["scheduler"])
    completed_epoch = checkpoint['epoch'] 
    loss = checkpoint['loss']

    best_val_loss = loss
    completed_steps = completed_epoch
    #scheduler.last_epoch = completed_epoch
    train_his_loss = torch.load("train_his_loss.pt", weights_only=False)
    validation_his_loss = torch.load("validation_his_loss.pt", weights_only=False)

    # best_val_loss = np.inf 
    # completed_steps = 0
    #train_his_loss, validation_his_loss = [], []

    train = True 
    pbar = tqdm(range(TRAINING_ITERATIONS - completed_steps))
    start_total = time.time()

    try:
        while train: 
            batch_training_loss = []
            batch_validation_loss = []
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

                batch_training_loss.append(loss.item())
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
                            batch_validation_loss.append(loss.item())
                    
                    epoch_training_loss_mean = np.mean(batch_training_loss)
                    epoch_valid_loss_mean = np.mean(batch_validation_loss)

                    # log to hist
                    train_his_loss.append(epoch_training_loss_mean)
                    validation_his_loss.append(epoch_valid_loss_mean)

                    # Save model if val loss decreases 
                    if epoch_valid_loss_mean < best_val_loss:
                        print(f"---Saving model---{time.time()}")
                        torch.save(model.state_dict(), "best_weights.pt")
                        torch.save({
                            'epoch': completed_steps,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            'loss': best_val_loss
                        }, "checkpoint.pt")
                                        
                        torch.save(train_his_loss, "train_his_loss.pt")
                        torch.save(validation_his_loss, "validation_his_loss.pt")

                        best_val_loss = epoch_valid_loss_mean

                    print("Training Loss:", epoch_training_loss_mean)
                    print("Validation Loss:", epoch_valid_loss_mean)

                    # Rest list 
                    batch_training_loss = []
                    batch_validation_loss = []

                    # Set Model to Training mode
                    model.train() 

                if completed_steps >= TRAINING_ITERATIONS: 
                    train = False 
                
                    print("Completed!", f"Total training time: {time.time() - start_total:.2f}s")
                    break
    except KeyboardInterrupt:
        print("Ctrl+C → saving checkpoint...")
        torch.save({
                    'epoch': completed_steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    'loss': best_val_loss
                }, "checkpoint.pt")
                                
        print("Saved")
    
if __name__ == "__main__":
    main()