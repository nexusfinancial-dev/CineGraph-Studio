"""
Integrated Neural TTS Voice Engine using Microsoft Edge Neural Voices.
Provides zero-configuration, high-fidelity character voice synthesis (.mp3)
for dialogue lines, character voice-overs, and cinematic narration without requiring ComfyUI.
"""

import os
import re
import asyncio
import edge_tts
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMFY_AUDIO_DIR = os.path.join(BASE_DIR, "data", "comfy_outputs", "audio")
os.makedirs(COMFY_AUDIO_DIR, exist_ok=True)

# Smart Voice Attribution based on character gender, species, and vocal persona
VOICE_MAP = {
    # Deep / Powerful / Fantasy
    "troll": "en-US-ChristopherNeural",
    "malak": "en-US-ChristopherNeural",
    "orc": "en-US-ChristopherNeural",
    "giant": "en-US-ChristopherNeural",
    # Rogue / Shadow / Goblin
    "goblin": "en-US-AriaNeural",
    "jez": "en-US-AriaNeural",
    "shadow": "en-US-AriaNeural",
    # Mage / Elf / Elegant Female
    "elf": "en-US-JennyNeural",
    "wynn": "en-US-JennyNeural",
    "fara": "en-US-AnaNeural",
    "hobkin": "en-US-AnaNeural",
    # Default Narrator
    "narrator": "en-US-RogerNeural",
}

def resolve_voice_for_character(character_name: str, voice_description: str = "", text: str = "") -> str:
    """Select the best neural voice actor based on character name and voice traits."""
    char_low = character_name.lower().strip()
    desc_low = voice_description.lower().strip()

    # 1. Check direct name match in VOICE_MAP
    for k, v in VOICE_MAP.items():
        if k in char_low:
            return v

    # 2. Check for Arabic text
    if any('\u0600' <= ch <= '\u06FF' for ch in text):
        if any(w in desc_low or w in char_low for w in ["female", "woman", "girl", "queen", "lady"]):
            return "ar-SA-ZariyahNeural"
        return "ar-SA-HamedNeural"

    # 3. Analyze voice description traits
    if any(w in desc_low or w in char_low for w in ["female", "feminine", "woman", "girl", "soprano", "velvety", "seductive", "breathy", "princess", "lady", "queen", "goddess", "priestess", "sorceress", "nymph", "heroine", "witch", "enchantress", "nydora"]):
        if any(w in desc_low for w in ["deep", "smoky", "contralto", "raspy", "authoritative"]):
            return "en-US-AriaNeural"
        if any(w in desc_low for w in ["young", "high", "cheerful", "playful"]):
            return "en-US-AnaNeural"
        return "en-US-JennyNeural"

    if any(w in desc_low for w in ["deep", "baritone", "gravelly", "gruff", "booming", "resonant", "troll", "monster", "heavy"]):
        return "en-US-ChristopherNeural"

    if any(w in desc_low for w in ["narrator", "exposition", "omniscient", "broadcast"]):
        return "en-US-RogerNeural"

    # Default male voice
    return "en-US-GuyNeural"

def clean_slug(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-]+', '_', s).strip('_').lower()

async def synthesize_speech_async(
    text: str,
    voice_name: str,
    output_path: str,
    rate: str = "+0%",
    pitch: str = "+0Hz"
):
    """Call Edge-TTS asynchronously and write audio file to disk."""
    clean_text = text.strip().strip('"').strip("'")
    # Clean instruction tags e.g. [Instruction: ...] before speaking
    clean_text = re.sub(r'\[Instruction:[^\]]*\]', '', clean_text).strip()
    
    tts = edge_tts.Communicate(clean_text, voice_name, rate=rate, pitch=pitch)
    await tts.save(output_path)

def generate_voice_audio(
    character_name: str,
    line_en: str,
    voice_description: str = "",
    shot_code: Optional[str] = None
) -> Dict[str, Any]:
    """Synchronous interface to synthesize voice line and return URL."""
    if not line_en or not line_en.strip():
        raise ValueError("Cannot synthesize speech for empty text line.")

    voice_name = resolve_voice_for_character(character_name, voice_description, line_en)
    
    char_slug = clean_slug(character_name)
    code_slug = clean_slug(shot_code) if shot_code else "dlg"
    rand_suffix = os.urandom(3).hex()
    filename = f"cine_{char_slug}_{code_slug}_{rand_suffix}.mp3"
    out_path = os.path.join(COMFY_AUDIO_DIR, filename)

    try:
        asyncio.run(synthesize_speech_async(line_en, voice_name, out_path))
        web_url = f"/audio/{filename}"
        return {
            "status": "success",
            "primary_audio": web_url,
            "filename": filename,
            "voice_used": voice_name,
            "character_name": character_name
        }
    except Exception as e:
        print(f"[TTS Engine Error] Failed to synthesize for {character_name}: {e}", flush=True)
        return {
            "status": "error",
            "error": str(e)
        }
