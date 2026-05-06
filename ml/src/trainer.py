# pip install torch torchaudio torchcodec 
# pip install transformers jiwer ipywidgets
import numpy as np 
from tqdm import tqdm # for displaying process bar
import torch 
import torch.nn as nn 
from torch import optim
from torch.utils.data import DataLoader
import yaml
from transformers import Wav2Vec2CTCTokenizer, get_cosine_schedule_with_warmup
import src.dataset.librispeech as librispeech
from src.dataset.librispeech import collate_fun
from shared.models import DeepSpeech2
from src.config import LIBRISPEECH_DIR, CONFIGS_DIR, SAVED_MODELS_DIR
import warnings
import time
import json

# import importlib
# importlib.reload(dataset)

warnings.simplefilter('ignore')

def compute_ctc_loss(model, batch, tokenizer, device):
    # input_lengths: seq_lens after conv layer: where is actual input, where is padding 
    logits, input_lengths = model(batch["input_values"].to(device), batch["seq_lens"])
    
    log_probs = nn.functional.log_softmax(logits, dim=-1) # dim=-1: apply softmax for the last dimension [batch_size, num_classes],
    log_probs = log_probs.transpose(0,1) # but CTC expect: [Time, batch_size, num_classes] => need to transpose two first dimensions

    #print(log_probs.shape) # => [batch_size, seq_lens, num_classes]

    # print(len(batch["labels"]), sum(batch["target_lengths"])) # the same number 


    return nn.functional.ctc_loss(
        log_probs=log_probs,
        targets=batch["labels"].to(device),
        input_lengths=input_lengths,
        target_lengths=batch["target_lengths"].to(device),
        blank=tokenizer.pad_token_id,
        reduction="mean",
    )


def main():

    # Load config
    with open(CONFIGS_DIR / "deepspeech2_config.yaml") as f:
        cfg = yaml.safe_load(f) 

    train_cfg = cfg["training"]
    audio_cfg = cfg["audio"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"] 

    DATASET_ROOT = str(LIBRISPEECH_DIR)
    BATCH_SIZE = train_cfg["batch_size"] 
    TRAINING_ITERATIONS = train_cfg["training_iterations"] 
    EVAL_ITERATIONS = train_cfg["eval_iterations"] 
    LEARNING_RATE = train_cfg["learning_rate"] 
    NUM_WORKERS = train_cfg["num_workers"] 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
    NUM_WARMUPS_STEPS = int(train_cfg["warmup_ratio"] * TRAINING_ITERATIONS)

    N_MELS = audio_cfg["n_mels"] 
    TOP_DB = audio_cfg["top_db"] 
    SAMPLING_RATE = audio_cfg["sampling_rate"] 
    MODEL_CONFIG = {**model_cfg, "n_mels": N_MELS, "top_db": TOP_DB, "sampling_rate": SAMPLING_RATE}

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(data_cfg["tokenizer_name"])

    ### Data loaders ###
    trainset = librispeech.LibrispeechDataset(path_to_data_root=DATASET_ROOT, include_splits=data_cfg["train_splits"], is_from_cached = False, is_augment=True, n_mels=N_MELS, top_db=TOP_DB, sampling_rate=SAMPLING_RATE)
    sampleset = librispeech.LibrispeechDataset(path_to_data_root=DATASET_ROOT, include_splits=data_cfg["val_splits"], is_from_cached = False)

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
    model = DeepSpeech2(conv_in_channels=MODEL_CONFIG["conv_in_channels"],
                        conv_out_channels=MODEL_CONFIG["conv_out_channels"],
                        rnn_hidden_size=MODEL_CONFIG["rnn_hidden_size"],
                        rnn_depth=MODEL_CONFIG["rnn_depth"],
                        rnn_dropout=MODEL_CONFIG.get("rnn_dropout", 0.3),
                        tokenizer=tokenizer
                        ).to(DEVICE)

    # numel: number of elements
    params = sum([p.numel() for p in model.parameters()])
    print("Total training Parameter: ", params)

    ### Optimizer ###
    optimizer = optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=NUM_WARMUPS_STEPS, num_training_steps=TRAINING_ITERATIONS)

    # model.load_state_dict(torch.load("best_weights.pt",weights_only=True)) # best_weights_512_3rnn, best_weights_128_2rnn
    checkpoint = torch.load(SAVED_MODELS_DIR /"checkpoint.pt", weights_only =False) # 

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint["scheduler"])
    completed_epoch = checkpoint['epoch'] 
    loss = checkpoint['loss']

    best_val_loss = loss
    completed_steps = completed_epoch
    #scheduler.last_epoch = completed_epoch
    train_his_loss = torch.load(SAVED_MODELS_DIR / "train_his_loss.pt", weights_only=False)
    validation_his_loss = torch.load(SAVED_MODELS_DIR / "validation_his_loss.pt", weights_only=False)

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
            
            for batch in trainloader: 
                # print(batch)

                loss = compute_ctc_loss(model, batch, tokenizer, DEVICE)

                # print(loss)

                loss.backward() # backpropagation
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=400) 
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

                            loss = compute_ctc_loss(model, batch, tokenizer, DEVICE)

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
                        

                        torch.save(model.state_dict(), SAVED_MODELS_DIR /"best_weights.pt")
                        with open(SAVED_MODELS_DIR / "config.json", 'w') as f: 
                            json.dump(MODEL_CONFIG, f, indent=2)
                        torch.save({
                            'epoch': completed_steps,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            'loss': best_val_loss
                        }, SAVED_MODELS_DIR / "checkpoint.pt")
                                        
                        torch.save(train_his_loss, SAVED_MODELS_DIR / "train_his_loss.pt")
                        torch.save(validation_his_loss, SAVED_MODELS_DIR / "validation_his_loss.pt")

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
                }, SAVED_MODELS_DIR / "checkpoint.pt")
                                
        print("Saved")


    
if __name__ == "__main__":
    main()