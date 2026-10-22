"""
Complete Scene Production Agent (GraphRAG + LangGraph)
Transforms raw literary narrative into complete Hollywood-style cinematic scene breakdowns:
Sluglines, dramatic objectives, active characters, action beats, dialogue flows, soundscapes,
and compiled ComfyUI/Flux cinematic keyframe prompts.
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
DATA_DIR = os.path.join(BASE_DIR, "data")
SCENE_BIBLE_PATH = os.path.join(DATA_DIR, "scene_bible.json")

# Local LM Studio Configuration
LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")

llm = model_resolver.get_chat_llm(request_timeout=60.0)

# ==========================================
# 1. Pydantic Strict Schemas
# ==========================================

class SceneDialogueBeat(BaseModel):
    speaker: str = Field(description="Name of the character speaking or thinking")
    line_en: str = Field(description="Verbatim English line spoken aloud or inner monologue")
    performance_instruction: str = Field(description="Acting directive: [Instruction: ...]")
    type: str = Field(default="spoken", description="'spoken' or 'voice_over'")

class SceneKeyframe(BaseModel):
    shot_type: str = Field(description="e.g. Wide Establishing Shot / Medium Two-Shot / Dramatic Close-Up")
    camera_angle: str = Field(description="e.g. Eye-level, Low angle, 24mm anamorphic wide, Dutch tilt")
    visual_prompt: str = Field(description="Production-ready ComfyUI / Flux visual prompt combining character, environment and lighting")
    description_en: str = Field(description="Brief cinematic description of what is visible in this keyframe")

class CompleteScene(BaseModel):
    id: str = Field(default_factory=lambda: f"sc_{uuid.uuid4().hex[:6]}")
    scene_number: int = Field(description="Sequential scene index (1, 2, 3...)")
    slugline: str = Field(description="Standard screenplay slugline (e.g. INT. REINCARNATION POD ROOM - DAWN)")
    location_name: str = Field(description="Canonical environment name")
    time_of_day: str = Field(default="DAY", description="DAWN / DAY / DUSK / NIGHT / CONTINUOUS")
    characters_present: List[str] = Field(description="Characters active on-screen in this scene")
    dramatic_objective: str = Field(description="Core dramatic tension, conflict, or goal of the scene")
    synopsis_en: str = Field(description="Concise 2-3 sentence overview of the entire scene progression")
    action_beats: List[str] = Field(description="Ordered sequence of visual actions, actor gestures, and blocking")
    dialogue_flow: List[SceneDialogueBeat] = Field(default=[], description="Sequential dialogue lines and voice-overs occurring in this scene")
    soundscape_and_foley: str = Field(description="Environmental audio ambiance, foley effects, and acoustic texture")
    keyframes: List[SceneKeyframe] = Field(default=[], description="Compiled shot keyframes for visual generation")
    raw_novel_excerpt: str = Field(description="Verbatim excerpt from novel confirming this scene")

class SceneBible(BaseModel):
    novel_title: str = ""
    scenes: List[CompleteScene] = []

class SceneAgentState(TypedDict):
    novel_text_path: str
    target_chapter_or_query: Optional[str]
    max_scenes: Optional[int]
    raw_scene_blocks: List[Dict[str, Any]]
    extracted_scenes: List[Dict[str, Any]]
    is_locked: bool
    bible_path: Optional[str]

# ==========================================
# 2. Narrative Segmentation Logic
# ==========================================

def clean_novel_text(text: str) -> str:
    """Normalize text encoding and line breaks."""
    text = text.replace("\r\n", "\n")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", " - ").replace("–", " - ")
    return text

def segment_novel_into_scenes(txt_path: str, max_scenes: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Intelligently segments the novel text into natural cinematic scenes based on:
    - Chapter markers and section headings
    - Scene transition delimiters (***, * * *, ---)
    - Temporal and spatial narrative shifts
    - Natural narrative pacing beats
    """
    if not os.path.exists(txt_path):
        return []

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    cleaned = clean_novel_text(content)
    raw_paragraphs = [p.strip() for p in cleaned.split("\n\n") if len(p.strip()) > 25]

    # Filter out TOC, patreon, and copyright boilerplate
    story_paragraphs = []
    found_start = False
    start_markers = ["chapter 1", "chapter one", "prologue", "part 1", "part one", "act 1"]
    for p in raw_paragraphs:
        p_low = p.lower()
        if any(m in p_low for m in start_markers):
            found_start = True
        if found_start or len(raw_paragraphs) < 300:
            if "oceanofpdf" not in p_low and "patreon" not in p_low and "all rights reserved" not in p_low:
                story_paragraphs.append(p)

    if not story_paragraphs:
        story_paragraphs = raw_paragraphs

    # Intelligent Narrative Scene Boundary Detection
    TIME_MARKERS = [
        r'^(the\s+next\s+morning|the\s+next\s+day|hours\s+later|meanwhile|later\s+that|by\s+dawn|the\s+following\s+morning)',
        r'^(that\s+evening|as\s+night\s+fell|morning\s+came|when\s+the\s+sun|after\s+several\s+hours)',
        r'^(eventually|finally|at\s+last|some\s+time\s+later|in\s+the\s+distance|outside|back\s+at)'
    ]
    transition_regex = re.compile('|'.join(TIME_MARKERS), re.IGNORECASE)

    scene_blocks = []
    current_paras = []

    for p in story_paragraphs:
        p_str = p.strip()
        is_break = False

        if p_str in ['***', '* * *', '---', '###']:
            is_break = True
        elif re.match(r'^(chapter\s+\d+|chapter\s+[a-z]+|prologue|epilogue|act\s+[ivx\d]+)', p_str, re.IGNORECASE):
            is_break = True
        elif len(current_paras) >= 12 and transition_regex.search(p_str):
            is_break = True
        elif len(current_paras) >= 28:
            is_break = True

        if is_break and current_paras:
            scene_blocks.append({
                "scene_number": len(scene_blocks) + 1,
                "paragraphs": current_paras
            })
            if max_scenes and len(scene_blocks) >= max_scenes:
                break
            current_paras = [p_str] if p_str not in ['***', '* * *', '---', '###'] else []
        else:
            if p_str not in ['***', '* * *', '---', '###']:
                current_paras.append(p_str)

    if current_paras and (not max_scenes or len(scene_blocks) < max_scenes):
        scene_blocks.append({
            "scene_number": len(scene_blocks) + 1,
            "paragraphs": current_paras
        })

    return scene_blocks

def infer_scene_metadata(scene_num: int, paragraphs: List[str], known_characters: Optional[List[str]] = None, known_locations: Optional[List[str]] = None) -> CompleteScene:
    """Analyze paragraphs and construct a rich, complete cinematic scene object dynamically."""
    full_block_text = "\n\n".join(paragraphs)
    text_low = full_block_text.lower()
    characters = known_characters or []
    locations_pool = known_locations or []

    # 1. Active Characters
    chars_present = []
    for ch in characters:
        if ch.lower() in text_low:
            chars_present.append(ch)
    if not chars_present:
        chars_present = [characters[0]] if characters else ["Protagonist"]

    # 2. Location & Slugline
    location = None
    for loc in locations_pool:
        if any(w.lower() in text_low for w in loc.split() if len(w) > 3):
            location = loc
            break

    if not location:
        if any(k in text_low for k in ["room", "bedroom", "apartment", "couch", "bed", "desk", "laptop", "chair"]):
            location = "Apartment Living Chamber"
            slugline = f"INT. APARTMENT - DAY"
            time_of_day = "DAY"
            soundscape = "Subdued interior room acoustics, soft ambient room tone, faint distant street tone"
        elif any(k in text_low for k in ["office", "studio", "building", "hall", "reception", "desk"]):
            location = "Production Studio & Facility"
            slugline = f"INT. FACILITY OFFICE - INTERIOR - DAY"
            time_of_day = "DAY"
            soundscape = "Commercial indoor tone, fluorescent ballast hum, muffled corridor footsteps"
        elif any(k in text_low for k in ["street", "outside", "car", "city", "park", "sky"]):
            location = "City Exterior & Streets"
            slugline = f"EXT. CITY DISTRICT - DAY"
            time_of_day = "DAY"
            soundscape = "Dynamic outdoor ambiance, traffic murmur, open-air acoustic resonance"
        else:
            location = locations_pool[0] if locations_pool else "Primary Scene Setting"
            slugline = f"INT. PRIMARY SCENE LOCATION - CONTINUOUS"
            time_of_day = "DAY"
            soundscape = "Natural room acoustics, subtle actor foley, ambient room tone"
    else:
        is_ext = any(w in location.lower() for w in ["exterior", "street", "skyline", "park", "outside"])
        time_of_day = "NIGHT" if any(w in text_low for w in ["night", "dark", "evening", "midnight"]) else "DAY"
        slugline = f"{'EXT.' if is_ext else 'INT.'} {location.upper()} - {time_of_day}"
        soundscape = f"Immersive cinematic acoustic environment of {location}, authentic spatial reverb, natural foley"

    # 3. Dramatic Objective & Synopsis
    narrative_sentences = []
    for p in paragraphs:
        if len(p) > 40 and not p.startswith('"') and not p.startswith('“'):
            for s in p.split("."):
                s_clean = s.strip()
                if len(s_clean) > 20 and not s_clean.lower().startswith("chapter"):
                    narrative_sentences.append(s_clean + ".")
                    if len(narrative_sentences) >= 3:
                        break
        if len(narrative_sentences) >= 3:
            break

    if narrative_sentences:
        dramatic_objective = f"Characters navigate key story developments: {narrative_sentences[0]}"
        synopsis = " ".join(narrative_sentences[:2])
    else:
        dramatic_objective = f"Dramatic interaction and pivotal story progression in {location}."
        synopsis = f"Characters navigate an unfolding sequence in {location}, contending with immediate narrative stakes."

    # 4. Action Beats
    action_beats = []
    for p in paragraphs:
        if len(p) > 50 and not p.startswith('"') and not p.startswith('“'):
            sentences = [s.strip() for s in p.split(".") if len(s.strip()) > 15]
            if sentences:
                action_beats.append(sentences[0] + ".")
        if len(action_beats) >= 4:
            break

    if not action_beats:
        action_beats = [
            f"Characters establish positions across {location}.",
            "Tension escalates as characters exchange revelatory dialogue.",
            "A shift in posture brings the scene toward a cinematic climax.",
            "The sequence transitions toward the following story beat."
        ]

    # 5. Dialogue Flow
    dialogue_flow = []
    dialogue_regex = re.compile(r'"([^"]{4,250})"')
    inner_thought_keywords = ["i wondered", "i thought", "occurred to me", "i realized"]

    for p in paragraphs:
        quotes = dialogue_regex.findall(p)
        if quotes:
            p_low = p.lower()
            speaker = chars_present[0] if chars_present else (characters[0] if characters else "Protagonist")
            for ch in (characters or chars_present):
                if ch.lower() in p_low:
                    speaker = ch
                    break

            for q in quotes:
                clean_q = q.strip()
                if len(clean_q.split()) >= 2:
                    dialogue_flow.append(SceneDialogueBeat(
                        speaker=speaker,
                        line_en=clean_q,
                        performance_instruction=f"[Instruction: Clear cinematic dialogue delivery, grounded character tone]",
                        type="spoken"
                    ))

        # Check for inner thoughts
        if any(kw in p.lower() for kw in inner_thought_keywords):
            for sent in re.split(r'[.!?]+', p):
                sent_str = sent.strip()
                if any(kw in sent_str.lower() for kw in inner_thought_keywords) and len(sent_str) > 20:
                    dialogue_flow.append(SceneDialogueBeat(
                        speaker=chars_present[0] if chars_present else (characters[0] if characters else "POV Protagonist"),
                        line_en=sent_str.replace('"', '').strip(),
                        performance_instruction="[Instruction: Intimate interior monologue, dry close-mic, reflective whisper, zero reverb]",
                        type="voice_over"
                    ))

        if len(dialogue_flow) >= 6:
            break

    # 6. Compiled Keyframes (ComfyUI / Flux / Krea Ready)
    keyframes = [
        SceneKeyframe(
            shot_type="Wide Establishing Shot",
            camera_angle="24mm anamorphic wide cinema lens, eye-level cinematic staging",
            visual_prompt=(
                f"Cinematic wide film still of {location}. Volumetric atmospheric lighting, detailed architectural depth, "
                f"cinematic color grading, 35mm film grain, anamorphic flare, photorealistic masterpiece, 8k resolution, no blur."
            ),
            description_en=f"Wide establishing overview of {location} capturing the visual mood and spatial scale."
        ),
        SceneKeyframe(
            shot_type="Medium Dramatic Shot",
            camera_angle="50mm prime portrait lens, shallow depth of field, over-the-shoulder framing",
            visual_prompt=(
                f"Cinematic film still, medium shot featuring {', '.join(chars_present[:2])} in {location}. "
                f"Authentic facial expressions, sharp focus on eyes, dramatic key lighting, photorealistic textures, movie still."
            ),
            description_en=f"Character interaction shot focusing on {', '.join(chars_present[:2])} during key dialogue beat."
        )
    ]

    excerpt = full_block_text[:350] + ("..." if len(full_block_text) > 350 else "")

    return CompleteScene(
        scene_number=scene_num,
        slugline=slugline,
        location_name=location,
        time_of_day=time_of_day,
        characters_present=chars_present,
        dramatic_objective=dramatic_objective,
        synopsis_en=synopsis,
        action_beats=action_beats,
        dialogue_flow=dialogue_flow,
        soundscape_and_foley=soundscape,
        keyframes=keyframes,
        raw_novel_excerpt=excerpt
    )

# ==========================================
# 3. Storage & Bible Persistence
# ==========================================

def load_scene_bible(bible_path: Optional[str] = None) -> Dict[str, Any]:
    """Load scene bible from disk."""
    target_path = bible_path or SCENE_BIBLE_PATH
    if os.path.exists(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"novel_title": "", "scenes": []}

def save_scene_bible(data: Dict[str, Any], bible_path: Optional[str] = None):
    """Persist scene bible to disk."""
    target_path = bible_path or SCENE_BIBLE_PATH
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def lock_scenes_into_bible(scenes: List[CompleteScene], novel_title: str = "", bible_path: Optional[str] = None) -> Dict[str, Any]:
    """Save extracted scenes into scene_bible.json."""
    current = load_scene_bible(bible_path=bible_path)
    current["novel_title"] = novel_title or current.get("novel_title", "Novel Production")
    
    existing_ids = {s.get("id") for s in current.get("scenes", [])}
    for sc in scenes:
        if sc.id not in existing_ids:
            current.setdefault("scenes", []).append(sc.model_dump())
            existing_ids.add(sc.id)

    # Re-index scene numbers
    for i, s in enumerate(current["scenes"]):
        s["scene_number"] = i + 1

    save_scene_bible(current, bible_path=bible_path)
    return current

# ==========================================
# 4. LangGraph State Machine
# ==========================================

def segment_novel_scenes_node(state: SceneAgentState) -> Dict[str, Any]:
    txt_path = state["novel_text_path"]
    max_sc = state.get("max_scenes")
    print(f"\n[Node: SegmentScenes] Parsing novel text for natural scene breaks: {txt_path}...", flush=True)
    blocks = segment_novel_into_scenes(txt_path, max_scenes=max_sc)
    print(f"[SegmentScenes] Auto-detected {len(blocks)} complete cinematic scenes from narrative flow.", flush=True)
    return {"raw_scene_blocks": blocks}

def extract_complete_scenes_node(state: SceneAgentState) -> Dict[str, Any]:
    blocks = state.get("raw_scene_blocks", [])
    txt_path = state.get("novel_text_path", "")
    print(f"[Node: ExtractCompleteScenes] Generating cinematic breakdown for {len(blocks)} scenes...", flush=True)

    # Discover characters and locations dynamically from manuscript
    try:
        from novel_production_graph import discover_novel_characters
        known_chars = discover_novel_characters(txt_path) if txt_path else []
    except Exception:
        known_chars = []

    try:
        from environment_production_graph import discover_novel_locations
        known_locs = discover_novel_locations(txt_path) if txt_path else []
    except Exception:
        known_locs = []

    scenes = []
    for blk in blocks:
        sc = infer_scene_metadata(blk["scene_number"], blk["paragraphs"], known_characters=known_chars, known_locations=known_locs)
        scenes.append(sc.model_dump())
        print(f"  * Scene {sc.scene_number}: {sc.slugline} ({len(sc.characters_present)} chars, {len(sc.dialogue_flow)} audio lines)", flush=True)

    return {"extracted_scenes": scenes}

def lock_scenes_node(state: SceneAgentState) -> Dict[str, Any]:
    extracted_data = state.get("extracted_scenes", [])
    scene_objects = [CompleteScene(**d) for d in extracted_data]
    novel_name = os.path.basename(state.get("novel_text_path", "")).replace(".txt", "").replace("_", " ").title()
    target_bible = state.get("bible_path") or SCENE_BIBLE_PATH
    lock_scenes_into_bible(scene_objects, novel_name, bible_path=target_bible)
    print(f"[Node: LockScenes] Successfully locked {len(scene_objects)} complete scenes into {target_bible}", flush=True)
    return {"is_locked": True}

def build_scene_pipeline_graph():
    """Build and compile the LangGraph StateGraph for complete scene extraction."""
    builder = StateGraph(SceneAgentState)
    
    builder.add_node("segment_scenes", segment_novel_scenes_node)
    builder.add_node("extract_scenes", extract_complete_scenes_node)
    builder.add_node("lock_scenes", lock_scenes_node)
    
    builder.add_edge(START, "segment_scenes")
    builder.add_edge("segment_scenes", "extract_scenes")
    builder.add_edge("extract_scenes", "lock_scenes")
    builder.add_edge("lock_scenes", END)
    
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# ==========================================
# 5. CLI Entry Point
# ==========================================

if __name__ == "__main__":
    txt_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "graphrag_workspace", "input", "solar_dragons_book1.txt")
    print(f"=== Starting Complete Scene Production Graph ===", flush=True)
    app = build_scene_pipeline_graph()
    initial_state: SceneAgentState = {
        "novel_text_path": txt_file,
        "target_chapter_or_query": None,
        "raw_scene_blocks": [],
        "extracted_scenes": [],
        "is_locked": False
    }
    config = {"configurable": {"thread_id": "cli_scene_extraction"}}
    result = app.invoke(initial_state, config=config)
    print("\n=== Scene Pipeline Execution Completed Successfully ===", flush=True)
    bible = load_scene_bible()
    print(f"Total Locked Complete Scenes: {len(bible.get('scenes', []))}")
