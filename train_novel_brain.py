"""
CineGraph Studio - Novel Brain LoRA Fine-Tuner (QLoRA)
Pure Verbatim Narrative Training (Zero Synthetic Hallucination)
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def clean_slug(title: str) -> str:
    import re
    return re.sub(r'[^a-zA-Z0-9_\-]+', '_', title).strip('_')

def resolve_novel_text(book_id: str) -> str:
    from run_agent import resolve_novel_text as runner_resolve
    return runner_resolve(book_id)

def prepare_verbatim_chunks(txt_path: str, chunk_size_words: int = 400, overlap_words: int = 50) -> List[str]:
    """Read verbatim novel text and partition into continuous narrative blocks."""
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Manuscript not found at: {txt_path}")
    
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Split into clean paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
    
    words = []
    for p in paragraphs:
        # Ignore boilerplate headers or page numbers
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

    print(f"[Dataset] Generated {len(chunks)} pure verbatim narrative chunks from {len(words)} words.")
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
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump({"status": "preparing", "progress": 5, "message": "Extracting clean manuscript..."}, f)

    # 1. Resolve text and generate verbatim chunks
    txt_path = resolve_novel_text(book_id)
    chunks = prepare_verbatim_chunks(txt_path)
    
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump({"status": "loading_model", "progress": 15, "message": f"Loading {base_model_name} in 4-bit QLoRA..."}, f)

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
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

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        warmup_ratio=0.05
    )

    with open(status_file, "w", encoding="utf-8") as f:
        json.dump({"status": "training", "progress": 30, "message": f"Training LoRA weights across {epochs} epochs on RTX GPU..."}, f)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=sft_config,
        processing_class=tokenizer
    )

    # 6. Execute Training
    train_result = trainer.train()
    
    # 7. Save Adapter
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump({"status": "saving", "progress": 90, "message": "Saving LoRA adapter..."}, f)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    elapsed = round(time.time() - start_time, 2)
    meta = {
        "status": "completed",
        "progress": 100,
        "book_id": book_id,
        "base_model": base_model_name,
        "epochs": epochs,
        "training_time_seconds": elapsed,
        "chunks_trained": len(chunks),
        "train_loss": round(float(train_result.training_loss), 4) if hasattr(train_result, "training_loss") else None,
        "adapter_dir": output_dir
    }

    with open(os.path.join(output_dir, "brain_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Success] Trained Novel Brain for '{book_id}' in {elapsed}s! Saved to: {output_dir}")
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
