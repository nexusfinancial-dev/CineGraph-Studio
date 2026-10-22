"""
Voiceprint & Audio Design Agent (Qwen3-TTS / CosyVoice)
Deterministic Voiceprint Extraction, 5 Acoustic Dimensions, and Original English Dialogue Lines
with line-by-line performance modifiers [Instruction: ...].
"""

import os
import sys
import json
import re
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

import model_resolver

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Local LM Studio Configuration
LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")

class CanonicalVoiceprint(BaseModel):
    character_name: str = Field(description="Name of the character")
    species: str = Field(description="Species / race affecting vocal anatomy (e.g. Hissfolk serpent woman, Plant Nymph, Human)")
    voiceprint_prompt: str = Field(
        description="Universal 5-dimension voice design prompt conforming to Qwen3-TTS / CosyVoice formula: "
                    "[Gender, Age & Persona] + [Vocal Pitch, Resonance & Texture] + [Pacing & Rhythmic Cadence] + "
                    "[Emotional Tone & Psychological State] + [Accent, Microphone Proximity & Acoustic Environment]"
    )
    vocal_timbre_and_pitch: str = Field(description="Register, pitch, vocal texture (e.g. low smoky contralto, serpentine sibilance, breathy soprano, resonant baritone)")
    cadence_and_pace: str = Field(description="Speech speed, rhythm, pausing, cadence")
    emotional_baseline: str = Field(description="Core psychological posture and attitude (e.g. seductive sarcasm, anxious vulnerability, authoritative)")
    accent_and_acoustics: str = Field(description="Accent and mic acoustics: dry close-mic studio recording, zero reverb, rich proximity effect")
    canonical_dialogue_samples: List[str] = Field(
        description="Original verbatim English dialogue lines from the novel, each prefixed with [Instruction: ...] performance tags. NEVER translated."
    )

def extract_character_dialogue_passages(char_name: str, txt_path: str, max_passages: int = 5) -> List[str]:
    """Search the novel text for direct spoken dialogue lines by the character."""
    if not os.path.exists(txt_path):
        return []

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    paragraphs = text.split("\n\n")
    dialogue_paragraphs = []
    
    char_lower = char_name.lower()
    for p in paragraphs:
        p_strip = p.strip()
        # Look for dialogue quotation marks and character mention
        if char_lower in p_strip.lower() and ('"' in p_strip or '“' in p_strip or '”' in p_strip):
            # Exclude paragraphs that are just descriptions without speech
            quotes = re.findall(r'["“](.*?)["”]', p_strip)
            if quotes and any(len(q.split()) >= 3 for q in quotes):
                dialogue_paragraphs.append(p_strip)
                if len(dialogue_paragraphs) >= max_passages:
                    break

    return dialogue_paragraphs

def get_precomputed_canonical_voiceprint(char_name: str, txt_path: Optional[str] = None) -> Optional[CanonicalVoiceprint]:
    """Deterministic canonical fallback voiceprints grounded 100% in the novel for key characters."""
    if txt_path and not ("solar" in txt_path.lower() or "dragons" in txt_path.lower()):
        return None
    c_lower = char_name.lower().strip()
    
    if "mamba" in c_lower:
        return CanonicalVoiceprint(
            character_name="Mamba",
            species="Hissfolk Serpentkin Woman",
            voiceprint_prompt=(
                "A buxom, confident female serpent woman in her mid-20s with a seductive, low-register contralto voice. "
                "Her vocal texture is smooth and velvety with a distinct, subtle reptilian rasp and a lingering sibilant "
                "elongation on 's' sounds that purrs gently in her throat. She speaks at an unhurried, sultry pace with "
                "deliberate, teasing pauses, projecting an attitude of playful dominance, dry sarcasm, and underlying warmth. "
                "Exotic subtle serpentine lilt, intimate close-mic studio recording, rich low-frequency warmth, zero background echo."
            ),
            vocal_timbre_and_pitch="Low velvety contralto with gentle reptilian vocal fry and subtle sibilant 's' elongation",
            cadence_and_pace="Unhurried, deliberate, sultry rhythm with amused teasing pauses",
            emotional_baseline="Confident, playful sarcasm, teasing dominance with protective loyalty",
            accent_and_acoustics="Intimate dry close-mic studio recording, zero reverb, rich proximity effect",
            canonical_dialogue_samples=[
                '[Instruction: Relaxed morning banter, amused dry smirk, soft purr on the final word] "Oh—I was going to sit on the couch. Guess I’ll just have to sit on your face."',
                '[Instruction: Sultry teasing whisper, slow unhurried delivery with lingering s-sounds] "Don’t tell me you’re getting nervous around a snake girl, Brock."',
                '[Instruction: Chuckle softly mid-sentence, playful defensive purr] "If anyone tries to touch what’s ours, they’ll find out how quick my venom works."'
            ]
        )
    
    elif "blossom" in c_lower:
        return CanonicalVoiceprint(
            character_name="Blossom",
            species="Adult Petite Plant Nymph",
            voiceprint_prompt=(
                "An 18-year-old adult petite plant nymph with a light, melodious soprano voice resembling delicate crystal wind chimes. "
                "Her vocal timbre is bright, airy, and crystalline with soft organic resonance and a musical fluttering cadence. "
                "She speaks with lively, eager pacing, shifting from enthusiastic curiosity to a defensive, sharp pout whenever mistaken for a sprout. "
                "Emotional baseline of vibrant innocence, earnest affection, and spirited stubborn pride. "
                "Neutral whimsical accent, crisp close-mic studio vocal capture, clean natural presence, zero reverb."
            ),
            vocal_timbre_and_pitch="Light crystalline soprano with airy, melodic botanical resonance",
            cadence_and_pace="Brisk, fluttering rhythm with spirited, expressive emphasis",
            emotional_baseline="Vibrant innocence, playful curiosity, spirited stubbornness when claiming adulthood",
            accent_and_acoustics="Crisp close-mic studio recording, clean presence, zero background echo",
            canonical_dialogue_samples=[
                '[Instruction: Pouting indignantly with hands on hips, sharp melodic emphasis] "I am eighteen! Stop calling me a cute little sprout!"',
                '[Instruction: Soft, wonder-filled whisper, breathless delight] "The sunlight on the balcony feels so warm... it makes my petals bloom."',
                '[Instruction: Playful giggle, bright chiming laugh] "Brock, look! My orchids opened up just because you smiled at me!"'
            ]
        )

    elif "brock" in c_lower:
        return CanonicalVoiceprint(
            character_name="Brock Clayton",
            species="Human Solar Dragon Champion",
            voiceprint_prompt=(
                "A rugged, athletic human male in his early 20s with a grounded, warm baritone voice. "
                "His vocal timbre is natural, firm, and resonant with subtle vocal fry on low trailing syllables. "
                "He speaks with steady, candid conversational pacing, frequently pausing with a bemused, self-aware chuckle "
                "as he reacts to the absurd fantasy world around him. Emotional baseline of adaptive composure, honest bemusement, "
                "and protective resolve. Neutral North American accent, dry close-mic broadcast recording, rich vocal chest resonance, zero reverb."
            ),
            vocal_timbre_and_pitch="Grounded warm baritone with natural chest resonance and subtle dry vocal fry",
            cadence_and_pace="Steady, conversational pacing with reflective, bemused pauses",
            emotional_baseline="Adaptive resilience, dry honest humor, protective groundedness",
            accent_and_acoustics="Neutral North American accent, dry studio close-mic recording, zero reverb",
            canonical_dialogue_samples=[
                '[Instruction: Bewildered, muttered to himself with a disoriented headshake] "Is this a weird-ass dream or did I really just wake up in a pink tube?"',
                '[Instruction: Dry, amused chuckle, affectionate exasperation] "Mamba, please... I’m just trying to make coffee without causing an incident."',
                '[Instruction: Firm, protective tone, steady eye contact] "Whatever’s coming out of that arena, we handle it together."'
            ]
        )

    return None

def extract_or_design_voiceprint(char_name: str, txt_path: str, book_id: Optional[str] = None, model_name: Optional[str] = None) -> CanonicalVoiceprint:
    """Extract or design voiceprint using local LLM with deterministic novel fallback."""
    precomputed = get_precomputed_canonical_voiceprint(char_name, txt_path=txt_path)
    if precomputed:
        return precomputed

    from pathlib import Path
    resolved_book = book_id or (Path(txt_path).stem if txt_path else None)
    active_model = model_name or model_resolver.get_best_local_uncensored_model(book_id=resolved_book)
    print(f"[Voiceprint] Designing acoustic profile for {char_name} via model '{active_model}'...", flush=True)

    # For dynamic novel characters not in the primary trio:
    dialogue_paragraphs = extract_character_dialogue_passages(char_name, txt_path, max_passages=4)
    evidence = "\n\n---\n\n".join(dialogue_paragraphs) if dialogue_paragraphs else f"Character name: {char_name}"

    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            base_url=LM_STUDIO_BASE_URL,
            api_key="lm-studio",
            model=active_model,
            temperature=0.0,
            request_timeout=60.0,
            max_retries=1
        )
        structured_llm = llm.with_structured_output(CanonicalVoiceprint)

        system_prompt = (
            "You are a master vocal director and acoustic sound designer for film and animation.\n"
            "Your task is to design a high-fidelity character voiceprint for Qwen3-TTS Voice Design / CosyVoice.\n\n"
            "FORMULA:\n"
            "[Gender, Age & Persona] + [Vocal Pitch, Resonance & Texture] + [Pacing & Rhythmic Cadence] + "
            "[Emotional Tone & Psychological State] + [Accent, Microphone Proximity & Acoustic Environment]\n\n"
            "CRITICAL: ALL DIALOGUES MUST BE IN THE NOVEL'S ORIGINAL LANGUAGE (ENGLISH). Do NOT translate dialogue lines."
        )

        user_prompt = f"Character: {char_name}\nNovel Dialogue Evidence:\n{evidence}\n\nExtract CanonicalVoiceprint JSON strictly adhering to schema."
        
        result: CanonicalVoiceprint = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        return result
    except Exception as e:
        print(f"[Voiceprint] Dynamic extraction fallback: {e}", flush=True)
        # Safe fallback
        return CanonicalVoiceprint(
            character_name=char_name,
            species="Novel Canonical Entity",
            voiceprint_prompt=(
                f"A distinct voice profile for {char_name}. Expressive vocal register with clear natural timbre, "
                "measured conversational cadence, authentic emotional baseline matching novel personality. "
                "Neutral accent, dry close-mic studio recording, clean presence, zero background echo."
            ),
            vocal_timbre_and_pitch="Clear, natural vocal register with balanced resonance",
            cadence_and_pace="Measured conversational cadence with natural expressive rhythm",
            emotional_baseline="Composed, character-consistent psychological posture",
            accent_and_acoustics="Dry close-mic studio recording, zero reverb",
            canonical_dialogue_samples=[
                f'[Instruction: Conversational, clear dialogue delivery] "Dialogue line from novel for {char_name}."'
            ]
        )

def lock_voiceprint_into_bible(char_name: str, voiceprint: CanonicalVoiceprint, bible_path: Optional[str] = None):
    """Lock the extracted voiceprint directly into character_bible.json."""
    if not bible_path:
        bible_path = os.environ.get("CHARACTER_BIBLE_PATH", os.path.join(BASE_DIR, "data", "character_bible.json"))
    bible = {}
    if os.path.exists(bible_path):
        try:
            with open(bible_path, "r", encoding="utf-8") as f:
                bible = json.load(f)
        except Exception:
            bible = {}

    key = char_name.lower().strip()
    if key not in bible:
        bible[key] = {
            "identity": {
                "name": voiceprint.character_name,
                "species": voiceprint.species
            },
            "is_locked": True
        }

    bible[key]["voiceprint"] = voiceprint.model_dump()
    bible[key]["has_voiceprint"] = True

    os.makedirs(os.path.dirname(bible_path), exist_ok=True)
    with open(bible_path, "w", encoding="utf-8") as f:
        json.dump(bible, f, indent=2, ensure_ascii=False)

    print(f"[Voiceprint] Successfully locked voiceprint for {char_name} into {bible_path}", flush=True)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "Protagonist"
    txt = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE_DIR, "graphrag_workspace", "input", "sample_novel.txt")
    print(f"=== Running Voiceprint Extraction for: {target} ===", flush=True)
    vp = extract_or_design_voiceprint(target, txt)
    lock_voiceprint_into_bible(target, vp)
    print("\nLocked Voiceprint:", flush=True)
    print(json.dumps(vp.model_dump(), indent=2), flush=True)
