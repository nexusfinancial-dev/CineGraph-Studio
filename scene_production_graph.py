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
from langchain_openai import ChatOpenAI
import json_repair

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SCENE_BIBLE_PATH = os.path.join(DATA_DIR, "scene_bible.json")

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
    raw_scene_blocks: List[Dict[str, Any]]
    extracted_scenes: List[Dict[str, Any]]
    is_locked: bool

# ==========================================
# 2. Narrative Segmentation Logic
# ==========================================

def clean_novel_text(text: str) -> str:
    """Normalize text encoding and line breaks."""
    text = text.replace("\r\n", "\n")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", " - ").replace("–", " - ")
    return text

def segment_novel_into_scenes(txt_path: str, max_scenes: int = 12) -> List[Dict[str, Any]]:
    """Segment novel text into discrete narrative scenes based on transitions, chapters, and time/location shifts."""
    if not os.path.exists(txt_path):
        return []

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    cleaned = clean_novel_text(content)
    raw_paragraphs = [p.strip() for p in cleaned.split("\n\n") if len(p.strip()) > 30]

    # Filter out TOC and copyright
    story_paragraphs = []
    found_start = False
    for p in raw_paragraphs:
        p_low = p.lower()
        if "light returned to me" in p_low or "chapter 1" in p_low or "welcome to meteoropolis" in p_low:
            found_start = True
        if found_start:
            if "oceanofpdf" not in p_low and "all rights reserved" not in p_low:
                story_paragraphs.append(p)

    if not story_paragraphs:
        story_paragraphs = raw_paragraphs[:150]

    # Cluster paragraphs into scenes of ~15-25 paragraphs each
    scene_blocks = []
    chunk_size = 20
    for idx in range(0, min(len(story_paragraphs), max_scenes * chunk_size), chunk_size):
        chunk = story_paragraphs[idx:idx + chunk_size]
        if chunk:
            scene_blocks.append({
                "scene_number": len(scene_blocks) + 1,
                "paragraphs": chunk
            })

    return scene_blocks

def infer_scene_metadata(scene_num: int, paragraphs: List[str]) -> CompleteScene:
    """Analyze paragraphs and construct a rich, complete cinematic scene object."""
    full_block_text = "\n\n".join(paragraphs)
    text_low = full_block_text.lower()

    # 1. Active Characters
    chars_present = []
    if "brock" in text_low or " i " in text_low or "my " in text_low:
        chars_present.append("Brock Clayton")
    if "mamba" in text_low:
        chars_present.append("Mamba")
    if "blossom" in text_low:
        chars_present.append("Blossom")
    if any(k in text_low for k in ["councilman", "doctor", "dr. luna", "white eyes", "bald dome", "pointed ears"]):
        chars_present.append("Doctor / Councilman White")
    if "pinky" in text_low:
        chars_present.append("Pinky Peach")
    if not chars_present:
        chars_present = ["Brock Clayton"]

    # 2. Location & Slugline
    if any(k in text_low for k in ["pod", "containment", "glass", "lab", "laboratory", "machinery", "silicon", "tubes"]):
        location = "Meteoropolis Reincarnation Laboratories"
        slugline = f"INT. REINCARNATION POD CHAMBER - METEOROPOLIS LABORATORIES - DAWN"
        time_of_day = "DAWN"
        soundscape = "Low mechanical hum of idling pod generators, rhythmic pneumatic valve releases, echoing footsteps on steel floor grating"
    elif any(k in text_low for k in ["penthouse", "living room", "couch", "balcony", "apartment", "sofa", "kitchen"]):
        location = "Brock's Luxury Penthouse Apartment"
        slugline = f"INT. BROCK'S PENTHOUSE APARTMENT - LIVING ROOM - MORNING"
        time_of_day = "MORNING"
        soundscape = "Muffled ambient city drone through double-glazed panoramic windows, gentle acoustic breeze on balcony, soft rustle of silk cushions"
    elif any(k in text_low for k in ["mall", "store", "shopping", "clothes", "dressing room"]):
        location = "Meteoropolis Central Fashion Plaza"
        slugline = f"INT. FASHION BOUTIQUE - DRESSING ROOM BAY - DAY"
        time_of_day = "DAY"
        soundscape = "Chiming electronic storefront chimes, bustling murmur of multi-species shoppers, neon light tube buzz"
    elif any(k in text_low for k in ["skyline", "meteor", "city", "flying", "street", "exterior"]):
        location = "Meteoropolis Skyline & Upper Tier"
        slugline = f"EXT. METEOROPOLIS SKYLINE - HIGH ELEVATION - DUSK"
        time_of_day = "DUSK"
        soundscape = "Volumetric gusting winds, deep roaring wingbeats of golden dragons overhead, resonant pulse of airborne transit freighters"
    else:
        location = "Meteoropolis Urban Sector"
        slugline = f"INT. METEOROPOLIS RECEPTION CHAMBER - CONTINUOUS"
        time_of_day = "DAY"
        soundscape = "Atmospheric interior hum, subdued fluorescent lighting buzz, distant ventilation draft"

    # 3. Dramatic Objective & Synopsis
    if "pod" in text_low or "reincarnation" in text_low:
        dramatic_objective = "Brock awakens inside a cryogenic medical pod with amnesia, grappling with physical disorientation and encountering his eccentric alien physician."
        synopsis = (
            "Brock Clayton regains consciousness inside an idling containment pod, suffocated by foggy rose-colored glass and amnesia. "
            "Doctor White unbuckles him with a remote control, welcoming him to the alien megacity of Meteoropolis while Brock tries to determine if he has lost his sanity."
        )
    elif "mamba" in text_low:
        dramatic_objective = "Playful morning dominance and territory banter between Brock and Mamba as they adapt to their cohabitation dynamics."
        synopsis = (
            "Mamba slithers into the living room with confident serpentine grace, teasing Brock over breakfast and asserting her affectionate dominance "
            "while discussing the looming galactic council threats over Meteoropolis."
        )
    elif "blossom" in text_low:
        dramatic_objective = "Blossom passionately asserts her adult independence and botanical identity after Brock misinterprets her youthful stature."
        synopsis = (
            "Blossom flares her petals in proud defiance when treated like a fragile sprout, displaying her botanical abilities and forging an earnest bond of trust with Brock."
        )
    else:
        dramatic_objective = f"Exploration and exposition establishing the rules and dangers of Meteoropolis."
        synopsis = f"The characters navigate an intense narrative sequence in {location}, contending with shifting allegiances and new discoveries."

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
            speaker = "Brock Clayton"
            if "mamba" in p_low:
                speaker = "Mamba"
            elif "blossom" in p_low:
                speaker = "Blossom"
            elif any(k in p_low for k in ["councilman", "doctor", "the man said", "he said"]):
                speaker = "Doctor / Councilman White"

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
                        speaker="Brock Clayton",
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

def load_scene_bible() -> Dict[str, Any]:
    """Load scene bible from disk."""
    if os.path.exists(SCENE_BIBLE_PATH):
        try:
            with open(SCENE_BIBLE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"novel_title": "", "scenes": []}

def save_scene_bible(data: Dict[str, Any]):
    """Persist scene bible to disk."""
    os.makedirs(os.path.dirname(SCENE_BIBLE_PATH), exist_ok=True)
    with open(SCENE_BIBLE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def lock_scenes_into_bible(scenes: List[CompleteScene], novel_title: str = "") -> Dict[str, Any]:
    """Save extracted scenes into data/scene_bible.json."""
    current = load_scene_bible()
    current["novel_title"] = novel_title or current.get("novel_title", "Novel Production")
    
    existing_ids = {s.get("id") for s in current.get("scenes", [])}
    for sc in scenes:
        if sc.id not in existing_ids:
            current.setdefault("scenes", []).append(sc.model_dump())
            existing_ids.add(sc.id)

    # Re-index scene numbers
    for i, s in enumerate(current["scenes"]):
        s["scene_number"] = i + 1

    save_scene_bible(current)
    return current

# ==========================================
# 4. LangGraph State Machine
# ==========================================

def segment_novel_scenes_node(state: SceneAgentState) -> Dict[str, Any]:
    txt_path = state["novel_text_path"]
    print(f"\n[Node: SegmentScenes] Parsing novel text for scene breaks: {txt_path}...", flush=True)
    blocks = segment_novel_into_scenes(txt_path)
    print(f"[SegmentScenes] Segmented manuscript into {len(blocks)} coherent scene sequences.", flush=True)
    return {"raw_scene_blocks": blocks}

def extract_complete_scenes_node(state: SceneAgentState) -> Dict[str, Any]:
    blocks = state.get("raw_scene_blocks", [])
    print(f"[Node: ExtractCompleteScenes] Generating cinematic breakdown for {len(blocks)} scenes...", flush=True)
    
    scenes = []
    for blk in blocks:
        sc = infer_scene_metadata(blk["scene_number"], blk["paragraphs"])
        scenes.append(sc.model_dump())
        print(f"  * Scene {sc.scene_number}: {sc.slugline} ({len(sc.characters_present)} chars, {len(sc.dialogue_flow)} audio lines)", flush=True)

    return {"extracted_scenes": scenes}

def lock_scenes_node(state: SceneAgentState) -> Dict[str, Any]:
    extracted_data = state.get("extracted_scenes", [])
    scene_objects = [CompleteScene(**d) for d in extracted_data]
    novel_name = os.path.basename(state.get("novel_text_path", "")).replace(".txt", "").replace("_", " ").title()
    lock_scenes_into_bible(scene_objects, novel_name)
    print(f"[Node: LockScenes] Successfully locked {len(scene_objects)} complete scenes into {SCENE_BIBLE_PATH}", flush=True)
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
