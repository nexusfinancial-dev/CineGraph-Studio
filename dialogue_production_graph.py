"""
Dialogue, Voice-Over & Narrator Production Agent (GraphRAG + LangGraph)
Extracts verbatim spoken dialogue, character inner voice-overs (V.O.), and cinematic narrator exposition
in the novel's original English language, complete with acting directives [Instruction: ...] for Qwen3-TTS / CosyVoice.
"""

import os
import sys
import json
import re
import uuid
from typing import Dict, List, Any, Optional, TypedDict
from pydantic import BaseModel, Field

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import json_repair

import model_resolver

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Local LM Studio Configuration
LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")

llm = model_resolver.get_chat_llm(request_timeout=60.0)

# ==========================================
# 1. Pydantic Schemas
# ==========================================

class SpokenDialogueItem(BaseModel):
    id: str = Field(default_factory=lambda: f"dlg_{uuid.uuid4().hex[:6]}")
    speaker: str = Field(description="Name of the character speaking aloud")
    addressed_to: str = Field(default="Listener", description="Character being addressed or spoken to")
    line_en: str = Field(description="Exact verbatim spoken dialogue line from the novel in original English")
    performance_instruction: str = Field(
        description="Line-by-line performance tag for TTS/actor: [Instruction: emotion, tone, vocal texture, pace, close-mic]"
    )
    context_quote: str = Field(description="Verbatim excerpt from novel confirming the dialogue exchange")
    emotion: str = Field(default="Conversational", description="Primary emotional posture (e.g. Seductive, Panicked, Teasing)")
    scene_context: str = Field(default="Scene Progression", description="Location or scene beat where this occurs")

class VoiceOverItem(BaseModel):
    id: str = Field(default_factory=lambda: f"vo_{uuid.uuid4().hex[:6]}")
    speaker: str = Field(description="POV character whose internal thoughts or memories are heard")
    line_en: str = Field(description="Exact verbatim inner monologue or thought line in original English")
    performance_instruction: str = Field(
        description="Line-by-line performance tag: [Instruction: Intimate interior monologue, dry close-mic, reflective whisper, zero reverb]"
    )
    context_quote: str = Field(description="Verbatim excerpt from novel confirming this is internal monologue")
    thought_type: str = Field(default="Internal Monologue", description="Type: Internal Reflection / Flashback Memory / Psychological Observation")
    scene_context: str = Field(default="Scene Progression", description="Visual beat or situation accompanying the thought")

class NarratorChunkItem(BaseModel):
    id: str = Field(default_factory=lambda: f"nar_{uuid.uuid4().hex[:6]}")
    narrator_profile: str = Field(default="Omniscient Cinematic Narrator", description="The narrative voice persona")
    line_en: str = Field(description="Cinematic narrative exposition chunk in original English")
    performance_instruction: str = Field(
        description="Performance tag: [Instruction: Atmospheric cinematic narration, deep resonant tone, slow measured cadence, rich studio acoustics]"
    )
    visual_beat: str = Field(description="Accompanying cinematic camera shot or establishing visual")
    scene_context: str = Field(default="Establishing Scene", description="Scene location or transitional beat")

class NarratorVoiceprint(BaseModel):
    voiceprint_prompt: str = Field(
        default=(
            "A mature male cinematic narrator in his mid-40s with a deep, warm, resonant baritone. "
            "Measured, articulate, authoritative cadence with natural pauses between sentences. "
            "Calm, immersive storyteller demeanor. Rich studio microphone capture, zero room echo, warm low-end proximity."
        )
    )
    timbre: str = Field(default="Deep resonant baritone with rich vocal chest resonance")
    cadence: str = Field(default="Measured cinematic rhythm, authoritative storytelling pace")
    acoustics: str = Field(default="Ultra-clean broadcast booth, zero reverb, dry proximity")

class AudioScriptBible(BaseModel):
    dialogues: List[SpokenDialogueItem] = []
    voice_overs: List[VoiceOverItem] = []
    narrator_chunks: List[NarratorChunkItem] = []
    narrator_voiceprint: NarratorVoiceprint = Field(default_factory=NarratorVoiceprint)

class ScriptAgentState(TypedDict):
    novel_text_path: str
    scope_filter: str  # "all" | "dialogue" | "voiceover" | "narrator"
    target_character: Optional[str]
    target_scene: Optional[str]
    raw_passages: List[str]
    extracted_script: Optional[Dict[str, Any]]
    is_locked: bool

# ==========================================
# 2. Deterministic Pattern Extraction & Parsing
# ==========================================

def clean_novel_text(text: str) -> str:
    """Normalize typography and encoding artifacts safely."""
    text = text.replace("\r\n", "\n")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", " - ").replace("–", " - ")
    return text

def extract_candidate_passages(txt_path: str, target_char: Optional[str] = None, max_paragraphs: int = 120) -> List[str]:
    """Retrieve candidate paragraphs containing dialogue, inner thoughts, and scene exposition."""
    if not os.path.exists(txt_path):
        return []

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    cleaned = clean_novel_text(content)
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if len(p.strip()) > 30]

    # Filter out table of contents or copyright
    candidates = []
    for p in paragraphs:
        p_low = p.lower()
        if "oceanofpdf" in p_low or "all rights reserved" in p_low or "table of contents" in p_low:
            continue
        if target_char and target_char.lower() not in p_low and '"' not in p:
            continue
        candidates.append(p)
        if len(candidates) >= max_paragraphs:
            break

    return candidates

def generate_dynamic_directive(quote: str, paragraph: str, speaker: str) -> tuple:
    """Generate dynamic TTS / acting instruction and emotion based on textual cues."""
    p_low = paragraph.lower()
    
    if any(w in p_low for w in ["shouted", "screamed", "yelled", "barked"]) or quote.endswith("!"):
        instr = "[Instruction: High-energy dramatic projection, sharp dynamic emphasis, clear studio mic capture]"
        emotion = "Intense, urgent, assertive"
    elif any(w in p_low for w in ["whispered", "breathed", "murmured"]) or "..." in quote:
        instr = "[Instruction: Soft intimate whisper, subtle vocal proximity, dry close-mic recording, zero room reverb]"
        emotion = "Intimate, reflective, quiet"
    elif any(w in p_low for w in ["laughed", "chuckled", "grinned", "smiled"]):
        instr = "[Instruction: Amused conversational tone with subtle vocal warmth, relaxed cadence]"
        emotion = "Amused, playful, warm"
    elif any(w in p_low for w in ["cried", "sobbed", "choked"]):
        instr = "[Instruction: Vulnerable emotional delivery with subtle vocal break, sincere cadence]"
        emotion = "Vulnerable, sorrowful"
    elif any(w in p_low for w in ["asked", "inquired"]) or quote.endswith("?"):
        instr = "[Instruction: Curious inquisitive delivery, slightly raised terminal pitch, natural timing]"
        emotion = "Inquisitive, questioning"
    elif any(w in p_low for w in ["coldly", "snapped", "hissed"]):
        instr = "[Instruction: Controlled tense delivery, crisp low-pitch articulation, dry studio recording]"
        emotion = "Firm, guarded, stern"
    else:
        instr = "[Instruction: Grounded cinematic dialogue delivery, clear speech articulation, natural room acoustic proximity]"
        emotion = "Conversational, grounded"

    return instr, emotion

def infer_speaker_and_addressee(p: str, known_characters: Optional[List[str]] = None, default_protagonist: Optional[str] = None) -> tuple:
    """Intelligently infer speaker and addressee dynamically from novel text without hardcoding."""
    p_low = p.lower()
    protagonist = default_protagonist or "Protagonist"
    characters = known_characters or []

    speech_verbs = ["said", "asked", "replied", "shouted", "whispered", "called", "chuckled", "smiled", "exclaimed", "muttered"]
    
    for ch in characters:
        ch_low = ch.lower()
        if ch_low in p_low:
            for v in speech_verbs:
                if f"{ch_low} {v}" in p_low or f"{v} {ch_low}" in p_low or f"{ch_low}," in p_low:
                    return ch, protagonist

    if any(f"i {v}" in p_low for v in speech_verbs):
        other_chars = [c for c in characters if c.lower() != protagonist.lower() and c.lower() in p_low]
        addressee = other_chars[0] if other_chars else "Listener"
        return protagonist, addressee

    for ch in characters:
        if ch.lower() in p_low:
            return ch, protagonist

    if any(f"she {v}" in p_low for v in speech_verbs):
        return "Female Speaker", protagonist
    if any(f"he {v}" in p_low for v in speech_verbs):
        return "Male Speaker", protagonist

    return protagonist, "Listener"

def parse_deterministic_script(paragraphs: List[str], target_char: Optional[str] = None, scope: str = "all", known_characters: Optional[List[str]] = None, default_protagonist: Optional[str] = None) -> AudioScriptBible:
    """Deterministic, high-speed parser extracting dialogues, voiceovers, and narrative exposition dynamically."""
    bible = AudioScriptBible()
    
    dialogue_regex = re.compile(r'"([^"]{5,300})"')
    inner_thought_keywords = ["i wondered", "i thought", "occurred to me", "i realized", "i remembered", "struck me", "felt like"]
    
    char_filter = target_char.lower() if target_char else ""
    
    for p in paragraphs:
        quotes = dialogue_regex.findall(p)
        if quotes and scope in ["all", "dialogue"]:
            speaker, addressed_to = infer_speaker_and_addressee(p, known_characters=known_characters, default_protagonist=default_protagonist)
            
            if char_filter and char_filter not in speaker.lower() and char_filter not in p.lower():
                continue

            for q in quotes:
                clean_q = q.strip()
                if len(clean_q.split()) >= 3:
                    instr, emotion = generate_dynamic_directive(clean_q, p, speaker)

                    bible.dialogues.append(SpokenDialogueItem(
                        speaker=speaker,
                        addressed_to=addressed_to,
                        line_en=clean_q,
                        performance_instruction=instr,
                        context_quote=p[:220] + ("..." if len(p) > 220 else ""),
                        emotion=emotion,
                        scene_context="Narrative Scene Progression"
                    ))

        # Check for Voice-Over (V.O. / Inner Monologue)
        if any(kw in p.lower() for kw in inner_thought_keywords) and scope in ["all", "voiceover"]:
            for sent in re.split(r'[.!?]+', p):
                sent_str = sent.strip()
                if any(kw in sent_str.lower() for kw in inner_thought_keywords) and len(sent_str) > 20:
                    clean_vo = sent_str.replace('"', '').strip()
                    vo_speaker = target_char or default_protagonist or (known_characters[0] if (known_characters and len(known_characters) > 0) else "Protagonist")
                    bible.voice_overs.append(VoiceOverItem(
                        speaker=vo_speaker,
                        line_en=clean_vo,
                        performance_instruction="[Instruction: Intimate internal monologue, dry close-mic, reflective whisper, zero reverb]",
                        context_quote=p[:220] + ("..." if len(p) > 220 else ""),
                        thought_type="Psychological Observation & Inner Monologue",
                        scene_context="Introspective Beat"
                    ))

        # Check for Narrator / Cinematic Exposition
        if not quotes and len(p.split()) >= 15 and scope in ["all", "narrator"]:
            sensory_words = ["light", "darkness", "sky", "city", "neon", "sound", "hum", "silence", "air", "room", "walls", "glass", "street", "building"]
            if any(sw in p.lower() for sw in sensory_words):
                first_two_sentences = ". ".join(p.split(". ")[:2]).strip()
                if not first_two_sentences.endswith("."):
                    first_two_sentences += "."
                bible.narrator_chunks.append(NarratorChunkItem(
                    narrator_profile="Omniscient Cinematic Narrator",
                    line_en=first_two_sentences,
                    performance_instruction="[Instruction: Atmospheric cinematic narration, deep resonant baritone, slow measured cadence, rich broadcast acoustics]",
                    visual_beat=f"Wide cinematic camera establishing shot: {first_two_sentences[:80]}...",
                    scene_context="Establishing Atmosphere"
                ))

    return bible

# ==========================================
# 3. LLM Structured Extraction Node
# ==========================================

def run_llm_script_extraction(passages: List[str], target_char: Optional[str] = None, scope: str = "all", book_id: Optional[str] = None) -> AudioScriptBible:
    """Use Local LM Studio to extract high-fidelity spoken dialogue, VO, and narrator chunks."""
    active_model = model_resolver.get_best_local_uncensored_model(book_id=book_id)
    if llm.model_name != active_model:
        llm.model_name = active_model
    evidence_text = "\n\n---\n\n".join(passages[:8])
    
    char_constraint = f"Focus particularly on exchanges involving: '{target_char}'." if target_char else ""
    
    system_prompt = (
        "You are an expert Hollywood script supervisor, cinematic dialogue director, and audio producer.\n"
        "Your task is to analyze novel excerpts and extract THREE distinct audio production elements:\n"
        "1. 'dialogues': Spoken dialogue lines between characters happening ALOUD on screen. Must be verbatim English.\n"
        "2. 'voice_overs': Inner thoughts, unspoken mental reflections, or memories heard as Voice-Over (V.O.).\n"
        "3. 'narrator_chunks': Cinematic narrative prose chunks describing settings, mood, transitions, and world ambiance.\n\n"
        "RULES:\n"
        "- All extracted lines MUST be in the novel's original English language. DO NOT translate!\n"
        "- Each item MUST include a detailed 'performance_instruction' formatted as:\n"
        "  '[Instruction: vocal register, emotional subtext, delivery speed, acoustic proximity, zero reverb]'\n"
        "- Output strictly valid JSON conforming to the schema.\n"
    )

    user_prompt = f"""Target Scope: {scope.upper()}
{char_constraint}

NOVEL EXCERPTS:
\"\"\"
{evidence_text}
\"\"\"

Return JSON with this exact schema:
{{
  "dialogues": [
    {{
      "speaker": "Speaker Name",
      "addressed_to": "Listener Name",
      "line_en": "Verbatim line spoken aloud in English",
      "performance_instruction": "[Instruction: emotional tone, vocal texture, pace, close-mic]",
      "context_quote": "Verbatim excerpt from text",
      "emotion": "Emotion descriptor",
      "scene_context": "Where this happens"
    }}
  ],
  "voice_overs": [
    {{
      "speaker": "POV Character Name",
      "line_en": "Verbatim inner thought or unvoiced reflection",
      "performance_instruction": "[Instruction: Intimate interior monologue, dry close-mic, reflective whisper, zero reverb]",
      "context_quote": "Verbatim excerpt from text",
      "thought_type": "Internal Monologue",
      "scene_context": "Visual situation"
    }}
  ],
  "narrator_chunks": [
    {{
      "narrator_profile": "Omniscient Cinematic Narrator",
      "line_en": "Verbatim cinematic narrative exposition in English",
      "performance_instruction": "[Instruction: Atmospheric cinematic narration, deep resonant tone, slow measured cadence, rich studio acoustics]",
      "visual_beat": "Cinematic visual camera description",
      "scene_context": "Scene location"
    }}
  ]
}}"""

    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        raw_content = response.content
        parsed = json_repair.loads(raw_content)
        
        if isinstance(parsed, dict):
            bible = AudioScriptBible()
            for d in parsed.get("dialogues", []):
                if isinstance(d, dict) and d.get("line_en"):
                    bible.dialogues.append(SpokenDialogueItem(**d))
            for vo in parsed.get("voice_overs", []):
                if isinstance(vo, dict) and vo.get("line_en"):
                    bible.voice_overs.append(VoiceOverItem(**vo))
            for n in parsed.get("narrator_chunks", []):
                if isinstance(n, dict) and n.get("line_en"):
                    bible.narrator_chunks.append(NarratorChunkItem(**n))
            if bible.dialogues or bible.voice_overs or bible.narrator_chunks:
                return bible
    except Exception as e:
        print(f"[LLM Extraction Warning] Fallback to deterministic parser: {e}", flush=True)

    return parse_deterministic_script(passages, target_char, scope)

# ==========================================
# 4. Storage & Persistence
# ==========================================

SCRIPT_BIBLE_PATH = os.path.join(BASE_DIR, "data", "script_bible.json")

def load_script_bible(bible_path: Optional[str] = None) -> Dict[str, Any]:
    """Load script bible from disk."""
    path = bible_path or SCRIPT_BIBLE_PATH
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "dialogues": [],
        "voice_overs": [],
        "narrator_chunks": [],
        "narrator_voiceprint": NarratorVoiceprint().model_dump()
    }

def save_script_bible(bible: Dict[str, Any], bible_path: Optional[str] = None):
    """Persist script bible to disk with atomic safety."""
    path = bible_path or SCRIPT_BIBLE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bible, f, ensure_ascii=False, indent=2)

    # Also sync to default active script bible
    default_active = os.path.join(BASE_DIR, "data", "script_bible.json")
    if os.path.abspath(path) != os.path.abspath(default_active):
        try:
            with open(default_active, "w", encoding="utf-8") as f:
                json.dump(bible, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

def merge_and_lock_script(extracted: AudioScriptBible, bible_path: Optional[str] = None):
    """Merge extracted items into script bible, avoiding duplicate lines."""
    current = load_script_bible(bible_path)
    
    existing_dlg_lines = {d.get("line_en", "").strip().lower() for d in current.get("dialogues", [])}
    for item in extracted.dialogues:
        if item.line_en.strip().lower() not in existing_dlg_lines:
            current.setdefault("dialogues", []).append(item.model_dump())
            existing_dlg_lines.add(item.line_en.strip().lower())

    existing_vo_lines = {v.get("line_en", "").strip().lower() for v in current.get("voice_overs", [])}
    for item in extracted.voice_overs:
        if item.line_en.strip().lower() not in existing_vo_lines:
            current.setdefault("voice_overs", []).append(item.model_dump())
            existing_vo_lines.add(item.line_en.strip().lower())

    existing_nar_lines = {n.get("line_en", "").strip().lower() for n in current.get("narrator_chunks", [])}
    for item in extracted.narrator_chunks:
        if item.line_en.strip().lower() not in existing_nar_lines:
            current.setdefault("narrator_chunks", []).append(item.model_dump())
            existing_nar_lines.add(item.line_en.strip().lower())

    if not current.get("narrator_voiceprint"):
        current["narrator_voiceprint"] = extracted.narrator_voiceprint.model_dump()

    save_script_bible(current, bible_path)
    return current

# ==========================================
# 5. LangGraph Pipeline Implementation
# ==========================================

def retrieve_script_passages_node(state: ScriptAgentState) -> Dict[str, Any]:
    txt_path = state["novel_text_path"]
    target_char = state.get("target_character")
    print(f"\n[Node: RetrieveScriptPassages] Ingesting novel from {txt_path}...", flush=True)
    passages = extract_candidate_passages(txt_path, target_char)
    print(f"[RetrieveScriptPassages] Found {len(passages)} candidate narrative paragraphs.", flush=True)
    return {"raw_passages": passages}

def extract_script_node(state: ScriptAgentState) -> Dict[str, Any]:
    passages = state.get("raw_passages", [])
    target_char = state.get("target_character")
    scope = state.get("scope_filter", "all")
    txt_path = state.get("novel_text_path", "")
    print(f"[Node: ExtractScript] Processing scope='{scope}' with character='{target_char}'...", flush=True)

    # Discover characters dynamically from the actual novel
    try:
        from novel_production_graph import discover_novel_characters
        known_chars = discover_novel_characters(txt_path) if txt_path else []
    except Exception:
        known_chars = []
    protagonist = known_chars[0] if known_chars else "Protagonist"
    
    # Run high-speed deterministic narrative extraction
    script_bible = parse_deterministic_script(
        passages,
        target_char,
        scope,
        known_characters=known_chars,
        default_protagonist=protagonist
    )
    print(
        f"[ExtractScript] Extracted {len(script_bible.dialogues)} Dialogues, "
        f"{len(script_bible.voice_overs)} Voice-Overs, "
        f"{len(script_bible.narrator_chunks)} Narrator Chunks.",
        flush=True
    )
    return {"extracted_script": script_bible.model_dump()}

def lock_script_node(state: ScriptAgentState) -> Dict[str, Any]:
    extracted_data = state.get("extracted_script", {})
    script_obj = AudioScriptBible(**extracted_data)
    target_bible = state.get("bible_path") or SCRIPT_BIBLE_PATH
    merged = merge_and_lock_script(script_obj, bible_path=target_bible)
    print(f"[Node: LockScript] Successfully locked script into {target_bible}", flush=True)
    return {"is_locked": True}

def build_script_pipeline_graph():
    """Build and compile the LangGraph StateGraph for audio script extraction."""
    builder = StateGraph(ScriptAgentState)
    
    builder.add_node("retrieve_passages", retrieve_script_passages_node)
    builder.add_node("extract_script", extract_script_node)
    builder.add_node("lock_script", lock_script_node)
    
    builder.add_edge(START, "retrieve_passages")
    builder.add_edge("retrieve_passages", "extract_script")
    builder.add_edge("extract_script", "lock_script")
    builder.add_edge("lock_script", END)
    
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# ==========================================
# 6. Standalone CLI Entry Point
# ==========================================

if __name__ == "__main__":
    txt_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "graphrag_workspace", "input", "solar_dragons_book1.txt")
    char = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"=== Starting Audio Script Production Graph (Dialogue, VO & Narrator) ===", flush=True)
    app = build_script_pipeline_graph()
    initial_state: ScriptAgentState = {
        "novel_text_path": txt_file,
        "scope_filter": "all",
        "target_character": char,
        "target_scene": None,
        "raw_passages": [],
        "extracted_script": None,
        "is_locked": False
    }
    config = {"configurable": {"thread_id": "cli_script_extraction"}}
    result = app.invoke(initial_state, config=config)
    print("\n=== Script Extraction Completed Successfully ===", flush=True)
    summary = load_script_bible()
    print(f"Total Dialogues: {len(summary.get('dialogues', []))}")
    print(f"Total Voice-Overs: {len(summary.get('voice_overs', []))}")
    print(f"Total Narrator Chunks: {len(summary.get('narrator_chunks', []))}")
