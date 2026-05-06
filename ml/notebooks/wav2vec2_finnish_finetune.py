# Fine-tune wav2vec2 on Finnish Common Voice
# Supports Google Colab with auto-save to Google Drive on disconnect
#
# Install (run once):
#   pip install transformers datasets jiwer soundfile librosa accelerate

import json
import os
import re
import shutil
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from jiwer import wer

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_MODEL    = "facebook/wav2vec2-large-xlsr-53"
OUTPUT_DIR    = "./wav2vec2-finnish"
VOCAB_FILE    = "./vocab_fi.json"
SAMPLING_RATE = 16_000

# Common Voice Finnish — requires HuggingFace login for some versions
# If you get an auth error: huggingface-cli login
DATASET_NAME = "mozilla-foundation/common_voice_11_0"
DATASET_LANG = "fi"

# ─── Google Drive setup ───────────────────────────────────────────────────────
# Mount Drive và sync checkpoint sau mỗi N steps
# Đặt USE_DRIVE = False nếu chạy trên Kaggle

USE_DRIVE  = True
DRIVE_DIR  = "/content/drive/MyDrive/wav2vec2-finnish"  # thư mục lưu trên Drive
SYNC_STEPS = 400                                          # sync mỗi 400 steps

def mount_drive():
    if not USE_DRIVE:
        return
    from google.colab import drive  # type: ignore[import]
    drive.mount("/content/drive")
    os.makedirs(DRIVE_DIR, exist_ok=True)
    print(f"Drive mounted. Checkpoints will sync to: {DRIVE_DIR}")

def sync_to_drive(local_dir: str):
    """Copy toàn bộ checkpoint mới nhất lên Google Drive."""
    if not USE_DRIVE:
        return
    try:
        shutil.copytree(local_dir, DRIVE_DIR, dirs_exist_ok=True)
        print(f"[Drive] Synced {local_dir} → {DRIVE_DIR}")
    except Exception as e:
        print(f"[Drive] Sync failed: {e}")

def resume_from_drive(local_dir: str):
    """Copy checkpoint từ Drive về local nếu có."""
    if not USE_DRIVE:
        return
    if os.path.exists(DRIVE_DIR) and os.listdir(DRIVE_DIR):
        shutil.copytree(DRIVE_DIR, local_dir, dirs_exist_ok=True)
        print(f"[Drive] Resumed from {DRIVE_DIR}")
    else:
        print("[Drive] No checkpoint found on Drive, starting fresh.")


class DriveSyncCallback(TrainerCallback):
    """Sync checkpoint lên Drive sau mỗi save step."""

    def on_save(self, args, state, control, **kwargs):  # noqa: ARG002
        sync_to_drive(args.output_dir)


# ─── 0. Mount Drive & resume ──────────────────────────────────────────────────

mount_drive()
resume_from_drive(OUTPUT_DIR)

# ─── 1. Load dataset ──────────────────────────────────────────────────────────

print("Loading dataset...")
train_data = load_dataset(DATASET_NAME, DATASET_LANG, split="train",    trust_remote_code=True)
val_data   = load_dataset(DATASET_NAME, DATASET_LANG, split="validation", trust_remote_code=True)

# Keep only audio + sentence columns
cols_to_remove = [c for c in train_data.column_names if c not in ("audio", "sentence")]
train_data = train_data.remove_columns(cols_to_remove)
val_data   = val_data.remove_columns(cols_to_remove)


# ─── 2. Clean transcripts ─────────────────────────────────────────────────────

CHARS_TO_REMOVE = r'[\,\?\.\!\-\;\:\"\"\%\'\"\`\']'

def clean_text(batch):
    batch["sentence"] = re.sub(CHARS_TO_REMOVE, "", batch["sentence"]).lower()
    return batch

train_data = train_data.map(clean_text)
val_data   = val_data.map(clean_text)


# ─── 3. Build vocabulary ──────────────────────────────────────────────────────

def extract_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = train_data.map(extract_chars, batched=True, batch_size=-1, keep_in_memory=True,
                              remove_columns=train_data.column_names)
vocab_val   = val_data.map(extract_chars,   batched=True, batch_size=-1, keep_in_memory=True,
                              remove_columns=val_data.column_names)

vocab_set = set(vocab_train["vocab"][0]) | set(vocab_val["vocab"][0])
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_set))}

# Replace space with visible token, add special tokens
vocab_dict["|"] = vocab_dict.pop(" ")
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open(VOCAB_FILE, "w") as f:
    json.dump(vocab_dict, f)

print(f"Vocab size: {len(vocab_dict)}")


# ─── 4. Processor ─────────────────────────────────────────────────────────────

tokenizer = Wav2Vec2CTCTokenizer(
    VOCAB_FILE,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=SAMPLING_RATE,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor.save_pretrained(OUTPUT_DIR)


# ─── 5. Preprocess audio ──────────────────────────────────────────────────────

train_data = train_data.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
val_data   = val_data.cast_column("audio",   Audio(sampling_rate=SAMPLING_RATE))

def preprocess(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

print("Preprocessing train...")
train_data = train_data.map(preprocess, remove_columns=train_data.column_names, num_proc=4)
print("Preprocessing val...")
val_data   = val_data.map(preprocess,   remove_columns=val_data.column_names,   num_proc=4)

# Filter clips > 10s to avoid OOM
MAX_INPUT_LENGTH = SAMPLING_RATE * 10
train_data = train_data.filter(lambda x: x < MAX_INPUT_LENGTH, input_columns=["input_length"])


# ─── 6. Data collator ─────────────────────────────────────────────────────────

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]}           for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")

        # Replace padding token id with -100 so it's ignored in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


# ─── 7. Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids    = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    return {"wer": wer(label_str, pred_str)}


# ─── 8. Model ─────────────────────────────────────────────────────────────────

print("Loading model...")
model = Wav2Vec2ForCTC.from_pretrained(
    BASE_MODEL,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

# Freeze feature extractor — only fine-tune transformer layers
model.freeze_feature_extractor()


# ─── 9. Training ──────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    group_by_length=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=30,
    gradient_checkpointing=True,
    fp16=True,                          # requires GPU
    save_steps=400,
    eval_steps=400,
    logging_steps=400,
    learning_rate=3e-4,
    warmup_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# Tự detect resume nếu có checkpoint trong OUTPUT_DIR
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    ckpts = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if ckpts:
        last_checkpoint = os.path.join(OUTPUT_DIR, sorted(ckpts, key=lambda x: int(x.split("-")[1]))[-1])
        print(f"Resuming from checkpoint: {last_checkpoint}")

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=processor.feature_extractor,
    callbacks=[DriveSyncCallback()] if USE_DRIVE else [],
)

print("Starting training...")
trainer.train(resume_from_checkpoint=last_checkpoint)

print("Saving final model...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
sync_to_drive(OUTPUT_DIR)
print("Done.")
