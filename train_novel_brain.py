"""
CineGraph Studio - Novel Brain LoRA Fine-Tuner (QLoRA)
Pure Verbatim Narrative Training (Zero Synthetic Hallucination)
"""

import os
import sys

# Ensure high-speed mirror and local-first fallbacks
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import json
import time
import argparse
import re
from typing import List, Dict, Any
import torch
from transformers import TrainerCallback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_HERMES_DIR = r"C:\Users\MiroAI\.cache\huggingface\hub\models--NousResearch--Hermes-3-Llama-3.1-8B\snapshots\896ea440e5a9e6070e3d8a2774daf2b481ab425b"

def clean_slug(title: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-]+', '_', title).strip('_')

def resolve_novel_text(book_id: str) -> str:
    from run_agent import resolve_novel_text as runner_resolve
    return runner_resolve(book_id)

def resolve_base_model_path(base_model_name: str) -> str:
    """Resolve model path prioritizing complete offline local weights."""
    if os.path.exists(LOCAL_HERMES_DIR) and os.path.exists(os.path.join(LOCAL_HERMES_DIR, "model.safetensors.index.json")):
        return LOCAL_HERMES_DIR
    return base_model_name

def safe_write_status(status_file: str, data: dict):
    try:
        temp_file = status_file + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if os.path.exists(temp_file):
            os.replace(temp_file, status_file)
    except Exception:
        pass

class NovelBrainProgressCallback(TrainerCallback):
    def __init__(self, status_file: str, epochs: int):
        self.status_file = status_file
        self.epochs = epochs
        self.start_train_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_train_time = time.time()
        max_steps = max(state.max_steps, 1)
        print(f"\n[Training Started] Total steps: {max_steps} across {self.epochs} epochs | Device: CUDA RTX GPU", flush=True)
        safe_write_status(self.status_file, {
            "status": "training",
            "progress": 25,
            "current_step": 0,
            "total_steps": max_steps,
            "epoch": 0.0,
            "loss": None,
            "eta": "calculating...",
            "message": f"Training started: Step 0/{max_steps}..."
        })

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        max_steps = max(state.max_steps, 1)
        # Scales smoothly from 25% to 90% during training steps
        pct = min(90, 25 + int((step / max_steps) * 65))
        
        # Calculate ETA
        eta_str = "calculating..."
        if self.start_train_time and step > 0:
            elapsed = time.time() - self.start_train_time
            sec_per_step = elapsed / step
            rem_sec = max(0, int((max_steps - step) * sec_per_step))
            m = rem_sec // 60
            s = rem_sec % 60
            eta_str = f"{m}m {s:02d}s"

        loss_val = None
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    loss_val = round(float(entry["loss"]), 4)
                    break

        epoch_val = round(float(state.epoch or 0.0), 2)
        loss_display = f"Loss: {loss_val}" if loss_val is not None else "Computing loss..."
        msg = f"Step {step}/{max_steps} (Epoch {epoch_val}) | {loss_display} | ETA: {eta_str}"

        safe_write_status(self.status_file, {
            "status": "training",
            "progress": pct,
            "current_step": step,
            "total_steps": max_steps,
            "epoch": epoch_val,
            "loss": loss_val,
            "eta": eta_str,
            "message": msg
        })

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = round(float(logs['loss']), 4)
            step = state.global_step
            max_steps = state.max_steps
            epoch = round(float(state.epoch or 0), 2)
            print(f"[Trainer Step {step}/{max_steps}] Epoch {epoch} | Loss: {loss}", flush=True)

def prepare_verbatim_chunks(txt_path: str, chunk_size_words: int = 400, overlap_words: int = 50) -> List[str]:
    """Read verbatim novel text and partition into continuous narrative blocks."""
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Manuscript not found at: {txt_path}")
    
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
    
    words = []
    for p in paragraphs:
        if p.lower().startswith("oceanofpdf") or p.lower().startswith("page ") or p.lower().startswith("table of contents"):
            continue
        words.extend(p.split())

    chunks = []
    i = 0
    step = chunk_size_words - overlap_words
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size_words])
        if len(chunk.strip()) > 100:
            chunks.append(chunk)
        i += step

    print(f"[Dataset] Generated {len(chunks)} pure verbatim narrative chunks from {len(words):,} words.", flush=True)
    return chunks

def train_novel_lora(
    book_id: str,
    base_model_name: str = "NousResearch/Hermes-3-Llama-3.1-8B",
    output_dir: str = None,
    epochs: int = 2,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1024
) -> Dict[str, Any]:
    start_time = time.time()
    slug = clean_slug(book_id)
    
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "data", "loras", slug)
    os.makedirs(output_dir, exist_ok=True)

    status_file = os.path.join(output_dir, "train_status.json")
    safe_write_status(status_file, {
        "status": "preparing",
        "progress": 5,
        "current_step": 0,
        "total_steps": 0,
        "epoch": 0.0,
        "loss": None,
        "eta": "preparing",
        "message": f"Extracting clean manuscript chunks for '{book_id}'..."
    })

    print(f"\n=======================================================", flush=True)
    print(f"[Novel Brain] Starting QLoRA Fine-Tuner", flush=True)
    print(f"[Novel Brain] Novel: {book_id}", flush=True)
    print(f"[Novel Brain] Base Model: {base_model_name}", flush=True)
    print(f"[Novel Brain] Epochs: {epochs} | Batch size: {batch_size}", flush=True)
    print(f"=======================================================\n", flush=True)

    # 1. Resolve text and generate verbatim chunks
    txt_path = resolve_novel_text(book_id)
    print(f"[Manuscript] Located manuscript at: {txt_path}", flush=True)
    chunks = prepare_verbatim_chunks(txt_path)
    
    safe_write_status(status_file, {
        "status": "loading_model",
        "progress": 15,
        "current_step": 0,
        "total_steps": 0,
        "epoch": 0.0,
        "loss": None,
        "eta": "loading model",
        "message": f"Loading Nous Hermes 3 (8B) in 4-bit NF4 QLoRA on GPU..."
    })

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    # 2. 4-bit Quantization Config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model_path = resolve_base_model_path(base_model_name)
    is_local = os.path.isdir(model_path)

    print(f"[Model] Loading tokenizer (source: {model_path}, local={is_local})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        local_files_only=is_local
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Model] Initializing 4-bit quantized base model weights on RTX GPU (local={is_local})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=is_local
    )
    model = prepare_model_for_kbit_training(model)
    vram_mb = model.get_memory_footprint() / 1024 / 1024
    print(f"[Model] Base model loaded into GPU VRAM successfully ({vram_mb:.1f} MB allocated).", flush=True)

    # 3. LoRA Configuration
    safe_write_status(status_file, {
        "status": "loading_model",
        "progress": 22,
        "current_step": 0,
        "total_steps": 0,
        "epoch": 0.0,
        "loss": None,
        "eta": "configuring lora",
        "message": "Configuring rank-16 LoRA adapter layers..."
    })
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    trainable_params, all_param = model.get_nb_trainable_parameters()
    print(f"[LoRA] Trainable parameters: {trainable_params:,} / {all_param:,} ({100 * trainable_params / all_param:.2f}%)", flush=True)

    # 4. Prepare Dataset
    dataset = Dataset.from_dict({"text": chunks})

    # 5. Training Arguments
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=2,
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        warmup_ratio=0.05
    )

    progress_cb = NovelBrainProgressCallback(status_file=status_file, epochs=epochs)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=sft_config,
        processing_class=tokenizer,
        callbacks=[progress_cb]
    )

    # 6. Execute Training
    train_result = trainer.train()
    
    # 7. Save Adapter
    safe_write_status(status_file, {
        "status": "saving",
        "progress": 92,
        "current_step": getattr(train_result, "global_step", epochs),
        "total_steps": getattr(train_result, "global_step", epochs),
        "epoch": float(epochs),
        "loss": round(float(train_result.training_loss), 4) if hasattr(train_result, "training_loss") else None,
        "eta": "saving adapter",
        "message": "Saving trained LoRA adapter and tokenizer..."
    })
    print(f"\n[Saving] Writing LoRA adapter weights to {output_dir}...", flush=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    elapsed = round(time.time() - start_time, 2)
    meta = {
        "status": "completed",
        "progress": 100,
        "current_step": getattr(train_result, "global_step", 0),
        "total_steps": getattr(train_result, "global_step", 0),
        "epoch": float(epochs),
        "book_id": book_id,
        "base_model": base_model_name,
        "epochs": epochs,
        "training_time_seconds": elapsed,
        "chunks_trained": len(chunks),
        "train_loss": round(float(train_result.training_loss), 4) if hasattr(train_result, "training_loss") else None,
        "adapter_dir": output_dir,
        "eta": "0m 00s",
        "message": f"Successfully trained Novel Brain in {elapsed}s! LoRA active."
    }

    with open(os.path.join(output_dir, "brain_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    safe_write_status(status_file, meta)

    print(f"[Success] Trained Novel Brain for '{book_id}' in {elapsed}s! Loss: {meta['train_loss']}", flush=True)
    return meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Novel Brain LoRA")
    parser.add_argument("--book-id", type=str, required=True, help="Title or ID of novel")
    parser.add_argument("--base-model", type=str, default="NousResearch/Hermes-3-Llama-3.1-8B", help="Base HF model")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    args = parser.parse_args()

    train_novel_lora(
        book_id=args.book_id,
        base_model_name=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
