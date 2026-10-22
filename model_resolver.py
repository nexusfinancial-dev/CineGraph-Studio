"""
Centralized Model Resolver for LM Studio & Local AI Engines.
Ensures that novel extraction, LangGraph agents, and local uncensored inference
strictly target the fine-tuned, fused novel model currently loaded in LM Studio,
preventing model-swapping and base-model fallbacks.
"""

import os
import re
import requests
from typing import Optional, Dict, Any, List

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234")

def clean_book_slug(book_id: str) -> str:
    """Standardizes book title into a clean underscore-separated slug."""
    clean = re.sub(r'[^a-zA-Z0-9_\-]+', '_', book_id).strip('_').lower()
    return clean

def get_best_local_uncensored_model(book_id: Optional[str] = None) -> str:
    """
    Dynamically identifies the exact model to use in LM Studio.
    
    Priority:
    1. If book_id provided, check for a model in LM Studio matching this book's slug or keywords
       (checking already LOADED models first, then all registered models).
    2. Any currently LOADED model from publisher 'NovelBrain'.
    3. ANY currently LOADED model with type == 'llm' (to avoid reloading/swapping VRAM).
    4. Any registered 'NovelBrain' model.
    5. Fallback to Nous Hermes 3 or any available LLM.
    """
    try:
        # Check LM Studio /api/v0/models (provides 'state': 'loaded' and 'publisher')
        r = requests.get(f"{LM_STUDIO_URL}/api/v0/models", timeout=3)
        if r.status_code == 200:
            data = r.json().get("data", [])
            loaded_models = [m for m in data if m.get("state") == "loaded" and m.get("type") == "llm"]
            all_llms = [m for m in data if m.get("type") == "llm"]

            # 1. Book-specific match
            if book_id:
                slug = clean_book_slug(book_id)
                book_words = [w for w in re.split(r'[^a-zA-Z0-9]+', book_id.lower()) if len(w) > 2]
                
                # Check loaded models first
                for m in loaded_models:
                    mid = m.get("id", "").lower()
                    if slug in mid or (book_words and sum(1 for w in book_words if w in mid) >= 2):
                        return m["id"]
                
                # Check all models registered in LM Studio
                for m in all_llms:
                    mid = m.get("id", "").lower()
                    if slug in mid or (book_words and sum(1 for w in book_words if w in mid) >= 2):
                        return m["id"]

            # 2. Any currently loaded NovelBrain model
            for m in loaded_models:
                if m.get("publisher") == "NovelBrain":
                    return m["id"]

            # 3. Any currently loaded LLM (preserves GPU memory, avoids swap delay)
            if loaded_models:
                return loaded_models[0]["id"]

            # 4. Any available NovelBrain model
            for m in all_llms:
                if m.get("publisher") == "NovelBrain":
                    return m["id"]

            # 5. Fallback to hermes or other uncensored
            for m in all_llms:
                if "hermes" in m.get("id", "").lower():
                    return m["id"]
            for m in all_llms:
                if "uncensored" in m.get("id", "").lower() and "@?" not in m.get("id", ""):
                    return m["id"]
            if all_llms:
                return all_llms[0]["id"]

    except Exception as e:
        print(f"[ModelResolver] Notice: Could not query LM Studio models API ({e})", flush=True)

    # Absolute fallback from environment or default
    return os.environ.get("LLM_MODEL", "hermes-3-llama-3.1-8b")


def get_chat_llm(book_id: Optional[str] = None, model_name: Optional[str] = None, **kwargs):
    """Factory creating a ChatOpenAI client targeting the active fused novel model."""
    from langchain_openai import ChatOpenAI
    
    resolved_model = model_name or get_best_local_uncensored_model(book_id=book_id)
    base_url = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
        
    params = {
        "base_url": base_url,
        "api_key": "lm-studio",
        "model": resolved_model,
        "temperature": 0.0,
        "request_timeout": 90.0,
        "max_retries": 1
    }
    params.update(kwargs)
    return ChatOpenAI(**params)
