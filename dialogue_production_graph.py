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
from langchain_openai import ChatOpenAI
import json_repair

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Local LM Studio Configuration
LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL", "gemma4-12b-qat-uncensored-hauhaucs-balanced@q4_k_m")

llm = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key="lm-studio",
    model=LLM_MODEL_NAME,
    temperature=0.0,
    request_timeout=60.0
)

# ==========================================
# 1. Pydantic Schemas
# ==========================================

class SpokenDialogueItem(BaseModel):
    id: str = Field(default_factory=lambda: f"dlg_{uuid.uuid4().hex[:6]}")
    speaker: str = Field(description="Name of the character speaking aloud (e.g. Mamba, Brock Clayton, Blossom)")
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
    speaker: str = Field(description="POV character whose internal thoughts or memories are heard (e.g. Brock Clayton)")
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

def infer_speaker_and_addressee(p: str, default_protagonist: str = "Brock Clayton") -> tuple:
    """Intelligently infer speaker and addressee from novel paragraph narrative markers."""
    p_low = p.lower()
    
    if "mamba" in p_low:
        if any(v in p_low for v in ["mamba said", "mamba purred", "mamba chuckled", "mamba smiled", "mamba hissed", "said mamba"]):
            return "Mamba", default_protagonist
        elif any(v in p_low for v in ["i said", "i asked", "i replied", "i called"]):
            return default_protagonist, "Mamba"
        return "Mamba", default_protagonist

    if "blossom" in p_low:
        if any(v in p_low for v in ["blossom said", "blossom squeaked", "blossom chimed", "blossom pouted"]):
            return "Blossom", default_protagonist
        elif any(v in p_low for v in ["i said", "i asked", "i replied"]):
            return default_protagonist, "Blossom"
        return "Blossom", default_protagonist

    if any(k in p_low for k in ["councilman", "dr. luna", "the man said", "he said"]):
        return "Doctor / Councilman", default_protagonist

    if any(v in p_low for v in ["i called", "i shouted", "i asked", "i said", "i muttered"]):
        return default_protagonist, "Listener"

    if any(v in p_low for v in ["she said", "she purred", "she chuckled"]):
        return "Female Companion", default_protagonist

    if any(v in p_low for v in ["he said", "he asked", "he barked"]):
        return "Male Companion", default_protagonist

    return default_protagonist, "Listener"

def parse_deterministic_script(paragraphs: List[str], target_char: Optional[str] = None, scope: str = "all") -> AudioScriptBible:
    """Deterministic, high-speed parser extracting dialogues, voiceovers, and narrative exposition."""
    bible = AudioScriptBible()
    
    dialogue_regex = re.compile(r'"([^"]{5,300})"')
    inner_thought_keywords = ["i wondered", "i thought", "occurred to me", "i realized", "i remembered", "struck me", "felt like"]
    
    char_filter = target_char.lower() if target_char else ""
    
    for p in paragraphs:
        # Check for Spoken Dialogues
        quotes = dialogue_regex.findall(p)
        if quotes and scope in ["all", "dialogue"]:
            speaker, addressed_to = infer_speaker_and_addressee(p)
            
            if char_filter and char_filter not in speaker.lower() and char_filter not in p.lower():
                continue

            for q in quotes:
                clean_q = q.strip()
                if len(clean_q.split()) >= 3:
                    if speaker == "Mamba":
                        instr = "[Instruction: Sultry contralto, teasing smile, playful dominance with lingering sibilant 's', dry close-mic studio recording, zero reverb]"
                        emotion = "Playful, teasing, confident"
                    elif speaker == "Blossom":
                        instr = "[Instruction: Bright crystalline soprano, airy fluttering cadence, spirited earnest tone, clean close-mic]"
                        emotion = "Earnest, vibrant, eager"
                    elif speaker == "Brock Clayton":
                        instr = "[Instruction: Conversational baritone, grounded male delivery, slightly bemused or dryly sarcastic, natural acoustic proximity]"
                        emotion = "Bewildered, conversational"
                    elif "Doctor" in speaker or "Councilman" in speaker:
                        instr = "[Instruction: Theatrical lower baritone, eccentric welcoming warmth, crisp close-mic studio capture]"
                        emotion = "Authoritative, welcoming, eccentric"
                    else:
                        instr = "[Instruction: Clear cinematic dialogue delivery, character-appropriate vocal texture, dry studio capture]"
                        emotion = "Conversational"

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
                    bible.voice_overs.append(VoiceOverItem(
                        speaker=target_char if target_char else "Brock Clayton",
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

def run_llm_script_extraction(passages: List[str], target_char: Optional[str] = None, scope: str = "all") -> AudioScriptBible:
    """Use Local LM Studio to extract high-fidelity spoken dialogue, VO, and narrator chunks."""
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

def load_script_bible() -> Dict[str, Any]:
    """Load script bible from disk."""
    if os.path.exists(SCRIPT_BIBLE_PATH):
        try:
            with open(SCRIPT_BIBLE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "dialogues": [],
        "voice_overs": [],
        "narrator_chunks": [],
        "narrator_voiceprint": NarratorVoiceprint().model_dump()
    }

def save_script_bible(bible: Dict[str, Any]):
    """Persist script bible to disk with atomic safety."""
    os.makedirs(os.path.dirname(SCRIPT_BIBLE_PATH), exist_ok=True)
    with open(SCRIPT_BIBLE_PATH, "w", encoding="utf-8") as f:
        json.dump(bible, f, ensure_ascii=False, indent=2)

def merge_and_lock_script(extracted: AudioScriptBible):
    """Merge extracted items into data/script_bible.json, avoiding duplicate lines."""
    current = load_script_bible()
    
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

    save_script_bible(current)
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
    print(f"[Node: ExtractScript] Processing scope='{scope}' with character='{target_char}'...", flush=True)
    
    # Run high-speed deterministic narrative extraction
    script_bible = parse_deterministic_script(passages, target_char, scope)
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
    merged = merge_and_lock_script(script_obj)
    print(f"[Node: LockScript] Successfully locked script into {SCRIPT_BIBLE_PATH}", flush=True)
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
