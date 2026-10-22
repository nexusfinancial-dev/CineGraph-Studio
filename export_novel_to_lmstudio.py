import os
import re
import sys
import time
import shutil
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_DIR = r"C:\Users\MiroAI\.cache\huggingface\hub\models--NousResearch--Hermes-3-Llama-3.1-8B\snapshots\896ea440e5a9e6070e3d8a2774daf2b481ab425b"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONVERT_SCRIPT = os.path.join(BASE_DIR, "tools", "llama.cpp", "convert_hf_to_gguf.py")
PYTHON_EXE = sys.executable

def get_slug(book_id: str) -> str:
    clean = re.sub(r'[\r\n\t]+', ' ', book_id).strip()
    slug = re.sub(r'[^a-zA-Z0-9_\-]+', '_', clean).strip('_')
    return slug

def export_novel_brain_to_lmstudio(book_id: str, outtype: str = "q8_0") -> dict:
    start_time = time.time()
    slug = get_slug(book_id)
    lora_dir = os.path.join(BASE_DIR, "data", "loras", slug)
    adapter_path = os.path.join(lora_dir, "adapter_model.safetensors")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"No trained LoRA found at {lora_dir}")

    # Destination directory in LM Studio models
    lmstudio_dir = os.path.expanduser(os.path.join("~", ".lmstudio", "models", "NovelBrain", slug))
    os.makedirs(lmstudio_dir, exist_ok=True)
    gguf_filename = f"{slug}-{outtype.upper()}.gguf"
    target_gguf_path = os.path.join(lmstudio_dir, gguf_filename)

    temp_merged_dir = os.path.join(BASE_DIR, "data", "temp_merged", slug)
    os.makedirs(temp_merged_dir, exist_ok=True)

    print(f"\n=======================================================", flush=True)
    print(f"[Merge & Export] Book: {book_id}", flush=True)
    print(f"[Merge & Export] Slug: {slug}", flush=True)
    print(f"[Merge & Export] Destination: {target_gguf_path}", flush=True)
    print(f"=======================================================\n", flush=True)

    # 1. Load Base Model in bfloat16 on CPU (using fast system RAM)
    print(f"[1/4] Loading base model weights on CPU ({time.time()-start_time:.1f}s)...", flush=True)
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, local_files_only=True)
    print(f"      Base model loaded into RAM in {time.time()-t0:.2f}s.", flush=True)

    # 2. Attach LoRA and Merge
    print(f"[2/4] Attaching LoRA adapter and fusing weights ({time.time()-start_time:.1f}s)...", flush=True)
    t1 = time.time()
    model = PeftModel.from_pretrained(base_model, lora_dir)
    merged_model = model.merge_and_unload()
    print(f"      Weights merged! Saving HuggingFace format to temp folder...", flush=True)
    merged_model.save_pretrained(temp_merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(temp_merged_dir)
    print(f"      Merged model written in {time.time()-t1:.2f}s.", flush=True)

    # Free memory
    del model
    del merged_model
    del base_model
    import gc
    gc.collect()

    # 3. Convert to GGUF using llama.cpp
    print(f"[3/4] Converting merged model to GGUF ({outtype}) via llama.cpp ({time.time()-start_time:.1f}s)...", flush=True)
    t2 = time.time()
    cmd = [
        PYTHON_EXE,
        CONVERT_SCRIPT,
        temp_merged_dir,
        "--outfile", target_gguf_path,
        "--outtype", outtype
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"STDERR: {res.stderr}", flush=True)
        raise RuntimeError(f"GGUF conversion failed: {res.stderr}")
    print(f"      GGUF file generated in {time.time()-t2:.2f}s!", flush=True)

    # Clean up temp merged directory
    shutil.rmtree(temp_merged_dir, ignore_errors=True)
    print(f"      Cleaned temporary files.", flush=True)

    # 4. Notify & Verify in LM Studio
    gguf_size_mb = round(os.path.getsize(target_gguf_path) / (1024 ** 2), 1)
    print(f"[4/4] Successfully exported to LM Studio models: {target_gguf_path} ({gguf_size_mb} MB)", flush=True)
    total_elapsed = round(time.time() - start_time, 2)
    print(f"=======================================================", flush=True)
    print(f"[COMPLETE] Novel Brain exported in {total_elapsed}s!", flush=True)
    print(f"=======================================================\n", flush=True)

    return {
        "status": "success",
        "book_id": book_id,
        "slug": slug,
        "gguf_path": target_gguf_path,
        "gguf_size_mb": gguf_size_mb,
        "total_seconds": total_elapsed
    }

if __name__ == "__main__":
    target_book = sys.argv[1] if len(sys.argv) > 1 else "A.J. Markam - Succubus - E01 - Succubus"
    export_novel_brain_to_lmstudio(target_book)
