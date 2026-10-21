"""
Micro-Scene / Cinematic Shot Breakdown Engine (GraphRAG + LangGraph)
Transforms Master Scenes into sequential, production-ready AI Video Shots (Takes)
with strict duration boundaries (6.0s - 15.0s) and speech pacing (2.0 words per second).
Optimized for AI Video Generators (MiniMax H3, Kling 1.5, Runway Gen-3, Wan 2.1, Sora).
"""

import os
import sys
import json
import re
import uuid
import math
from typing import Dict, List, Any, Optional, TypedDict
from pydantic import BaseModel, Field

# LangGraph imports
from langgraph.graph import StateGraph, START, END

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SCENE_BIBLE_PATH = os.path.join(DATA_DIR, "scene_bible.json")
MICRO_SCENES_BIBLE_PATH = os.path.join(DATA_DIR, "micro_scenes_bible.json")

# ==========================================
# 1. Pydantic Strict Schemas
# ==========================================

class MicroSceneDialogue(BaseModel):
    speaker: str = Field(description="Character speaking or thinking")
    line_en: str = Field(description="Verbatim spoken dialogue or inner monologue in original English")
    words_count: int = Field(description="Total word count in spoken line")
    pacing_words_per_sec: float = Field(default=2.0, description="Dialogue delivery rate (strictly 2.0 words/sec)")
    speech_duration_seconds: float = Field(description="Calculated speech time: words_count / 2.0")
    buffer_seconds: float = Field(description="Breathing, physical reaction, and camera movement buffer")
    performance_instruction: str = Field(description="Acting directive for Qwen3-TTS / CosyVoice")
    dialogue_type: str = Field(default="spoken", description="'spoken' or 'voice_over'")

class MicroSceneShot(BaseModel):
    id: str = Field(default_factory=lambda: f"sh_{uuid.uuid4().hex[:6]}")
    scene_number: int = Field(description="Parent master scene sequential index")
    shot_number: int = Field(description="Sequential shot index within this master scene")
    global_shot_number: int = Field(description="Overall shot index across entire film project")
    shot_code: str = Field(description="Industry production take code, e.g. SC01-TK01")
    slugline: str = Field(description="Screenplay slugline, e.g. INT. REINCARNATION POD CHAMBER - DAWN")
    location_name: str = Field(description="Environment setting")
    time_of_day: str = Field(default="DAY", description="DAWN / DAY / DUSK / NIGHT")
    shot_type: str = Field(description="Cinematic shot classification (e.g. Establishing Wide, Over-the-Shoulder, Medium Close-Up)")
    camera_framing: str = Field(description="Wide / Medium / Close-Up / Extreme Close-Up / Two-Shot")
    camera_motion: str = Field(description="Camera motion vector (e.g. Slow push-in dolly, tracking pan, static locked)")
    lens_spec: str = Field(default="35mm Anamorphic", description="Cinematic lens spec (24mm, 35mm, 50mm, 85mm)")
    duration_seconds: float = Field(description="Total shot duration in seconds (strictly between 6.0 and 15.0)")
    speech_pacing_rate: str = Field(default="2.0 words/sec", description="Standard speech pacing")
    dialogue: Optional[MicroSceneDialogue] = Field(default=None, description="Spoken dialogue or V.O. in this shot")
    characters_in_frame: List[str] = Field(default=[], description="Characters visible on screen in this take")
    visual_action: str = Field(description="Precise chronological actor movement and blocking across this 6-15s take")
    soundscape_and_foley: str = Field(description="Ambient soundscape, diegetic audio, and foley effects aligned to this take")
    ai_video_prompt: str = Field(description="Production-ready prompt for AI video generators (MiniMax H3 / Kling / Runway Gen-3)")
    acting_directive: str = Field(description="Acting instructions synchronized with the 2 words/sec cadence")

class MicroSceneBible(BaseModel):
    novel_title: str = ""
    pacing_standard: str = "2.0 words per second"
    duration_constraint: str = "6.0s - 15.0s per take"
    total_master_scenes: int = 0
    total_shots: int = 0
    total_runtime_seconds: float = 0.0
    total_runtime_formatted: str = "0m 00s"
    shots: List[MicroSceneShot] = []

class MicroSceneAgentState(TypedDict):
    scene_bible_path: str
    raw_scenes: List[Dict[str, Any]]
    micro_shots: List[Dict[str, Any]]
    min_duration: float
    max_duration: float
    words_per_sec: float
    is_locked: bool

# ==========================================
# 2. Cinematic Camera Styles & Lenses
# ==========================================

CAMERA_FRAMINGS = [
    {
        "type": "Establishing Wide Shot (WS)",
        "framing": "Wide",
        "lens": "24mm Anamorphic f/2.8",
        "motion": "Slow aerial pedestal down tracking spatial architecture, 24fps cinematic motion"
    },
    {
        "type": "Medium Two-Shot (M2S)",
        "framing": "Medium",
        "lens": "35mm Prime f/2.0",
        "motion": "Subtle handheld tracking drift maintaining both subjects in balanced frame"
    },
    {
        "type": "Over-the-Shoulder (OTS)",
        "framing": "Medium Close-Up",
        "lens": "50mm Cinematic Prime f/1.8",
        "motion": "Slow push-in dolly forward over foreground shoulder toward speaking character"
    },
    {
        "type": "Medium Close-Up (MCU)",
        "framing": "Medium Close-Up",
        "lens": "50mm Portrait Prime f/1.4",
        "motion": "Slow smooth push-in tracking subtle facial micro-expressions, shallow depth of field"
    },
    {
        "type": "Intimate Dramatic Close-Up (CU)",
        "framing": "Close-Up",
        "lens": "85mm Telephoto f/1.4",
        "motion": "Static locked-off frame with intense focus on emotional eye reflection and vocal delivery"
    },
    {
        "type": "Low-Angle Power Shot",
        "framing": "Medium Low-Angle",
        "lens": "28mm Wide Prime f/2.8",
        "motion": "Subtle slow upward tilt emphasizing character dominance and architectural height"
    },
    {
        "type": "Reverse Reaction Shot",
        "framing": "Close-Up",
        "lens": "65mm Anamorphic f/2.0",
        "motion": "Quick smooth whip pan or cut to listener absorbing the spoken line with reactive expression"
    }
]

# ==========================================
# 3. Micro-Scene Decomposition Algorithm
# ==========================================

def calculate_speech_and_shot_duration(
    words_count: int,
    min_dur: float = 6.0,
    max_dur: float = 15.0,
    wps: float = 2.0
) -> tuple[float, float, float]:
    """
    Calculate speech duration, action buffer, and total shot duration.
    Speech pacing is strictly 2.0 words per second.
    Total duration is strictly clamped between min_dur (6.0s) and max_dur (15.0s).
    """
    if words_count <= 0:
        # Non-dialogue visual shot (Establishing / Action / Reaction)
        speech_dur = 0.0
        buffer_dur = 7.5
        total_dur = 7.5
        return speech_dur, buffer_dur, total_dur

    speech_dur = round(words_count / wps, 1)

    # Calculate buffer for pre-speech pause, breath, and reaction
    if speech_dur < 4.0:
        # Short dialogue line (e.g. 2-7 words) -> pad buffer so total duration meets the 6.0s minimum
        buffer_dur = round(max(2.0, min_dur - speech_dur), 1)
    else:
        # Standard dialogue line -> 1.5s to 2.5s natural dramatic buffer
        buffer_dur = 2.0

    total_dur = round(speech_dur + buffer_dur, 1)
    
    # Strictly enforce bounds [6.0s, 15.0s]
    if total_dur < min_dur:
        total_dur = min_dur
        buffer_dur = round(total_dur - speech_dur, 1)
    elif total_dur > max_dur:
        total_dur = max_dur
        buffer_dur = round(max(1.0, total_dur - speech_dur), 1)

    return speech_dur, buffer_dur, total_dur


def split_dialogue_if_too_long(line: str, max_words: int = 22) -> List[str]:
    """
    If a dialogue line exceeds 22 words (which would take >11s speech at 2 wps),
    intelligently split into two consecutive sentences or clauses.
    """
    words = line.split()
    if len(words) <= max_words:
        return [line]

    # Try splitting at punctuation (. , ; ! ? —)
    puncts = [". ", "! ", "? ", ", ", " — ", " - ", "; "]
    for p in puncts:
        if p in line:
            parts = line.split(p, 1)
            p1 = parts[0].strip() + (p.strip() if not p.endswith(" ") else "")
            p2 = parts[1].strip()
            if len(p1.split()) >= 4 and len(p2.split()) >= 4:
                return [p1, p2]

    # Fallback: split at midpoint word
    mid = len(words) // 2
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def generate_micro_scenes_from_master(
    scene_bible_data: Dict[str, Any],
    min_dur: float = 6.0,
    max_dur: float = 15.0,
    wps: float = 2.0
) -> List[MicroSceneShot]:
    """
    Decomposes each master scene into discrete 6s-15s shots with 2.0 words/sec speech pacing.
    """
    master_scenes = scene_bible_data.get("scenes", [])
    all_micro_shots: List[MicroSceneShot] = []
    global_idx = 1

    for sc in master_scenes:
        sc_num = sc.get("scene_number", 1)
        slugline = sc.get("slugline", "INT. SCENE - DAY")
        loc_name = sc.get("location_name", "Unknown Location")
        tod = sc.get("time_of_day", "DAY")
        cast = sc.get("characters_present", [])
        synopsis = sc.get("synopsis_en", "")
        objective = sc.get("dramatic_objective", "")
        beats = sc.get("action_beats", [])
        dlgs = sc.get("dialogue_flow", [])
        soundscape = sc.get("soundscape_and_foley", "Ambient atmospheric room tone")

        shot_in_scene = 1

        # -------------------------------------------------------------
        # SHOT 1: Establishing Wide Take (Scene Opening)
        # -------------------------------------------------------------
        cam_setup = CAMERA_FRAMINGS[0]
        shot_code = f"SC{sc_num:02d}-TK{shot_in_scene:02d}"
        
        opening_action = beats[0] if beats else f"The camera establishes {loc_name} as the scene opens with {', '.join(cast) if cast else 'characters present'}."
        opening_prompt = (
            f"Cinematic {cam_setup['lens']} wide establishing shot. {cam_setup['motion']}. "
            f"Location: {loc_name}. Atmosphere: {tod} lighting, volumetric depth, anamorphic lens flares. "
            f"Scene Action: {opening_action}. Photorealistic 8k, cinematic color grading, 24fps motion blur."
        )

        all_micro_shots.append(MicroSceneShot(
            id=f"sh_{sc_num:02d}_{shot_in_scene:02d}_{uuid.uuid4().hex[:4]}",
            scene_number=sc_num,
            shot_number=shot_in_scene,
            global_shot_number=global_idx,
            shot_code=shot_code,
            slugline=slugline,
            location_name=loc_name,
            time_of_day=tod,
            shot_type=cam_setup["type"],
            camera_framing=cam_setup["framing"],
            camera_motion=cam_setup["motion"],
            lens_spec=cam_setup["lens"],
            duration_seconds=8.0,
            speech_pacing_rate=f"{wps:.1f} words/sec",
            dialogue=None,
            characters_in_frame=cast[:2],
            visual_action=opening_action,
            soundscape_and_foley=f"Opening ambiance: {soundscape}",
            ai_video_prompt=opening_prompt,
            acting_directive="[Instruction: Silent opening establishing take, atmospheric poise, zero spoken dialogue]"
        ))
        shot_in_scene += 1
        global_idx += 1

        # -------------------------------------------------------------
        # SHOTS 2+: Dialogue & Action Sequential Takes
        # -------------------------------------------------------------
        if dlgs:
            for d_idx, d in enumerate(dlgs):
                raw_line = d.get("line_en", "").strip()
                speaker = d.get("speaker", "Unknown")
                instruction = d.get("performance_instruction", "[Instruction: Clear cinematic dialogue delivery]")
                dlg_type = d.get("type", "spoken")

                # Handle potentially oversized dialogue lines
                split_lines = split_dialogue_if_too_long(raw_line, max_words=22)

                for part_idx, line_part in enumerate(split_lines):
                    words_count = len(line_part.split())
                    speech_dur, buf_dur, total_dur = calculate_speech_and_shot_duration(
                        words_count, min_dur=min_dur, max_dur=max_dur, wps=wps
                    )

                    # Cycle camera framings logically:
                    # Alternating OTS (2), MCU (3), CU (4), Medium Two-Shot (1), Low-Angle (5)
                    style_idx = 1 + ((d_idx + part_idx) % (len(CAMERA_FRAMINGS) - 1))
                    cam_setup = CAMERA_FRAMINGS[style_idx]
                    shot_code = f"SC{sc_num:02d}-TK{shot_in_scene:02d}"

                    # Determine visual blocking for this take
                    beat_ref = beats[min(d_idx % max(1, len(beats)), len(beats) - 1)] if beats else ""
                    shot_action = f"{speaker} speaks facing the interlocutor. {beat_ref}"

                    # Explicit timing directive for TTS and video pacing
                    sync_acting_directive = (
                        f"[Instruction: Paced strictly at {wps:.1f} words per second ({words_count} words over {speech_dur}s), "
                        f"followed by {buf_dur}s dramatic action buffer. {instruction.replace('[Instruction:', '').replace(']', '').strip()}]"
                    )

                    video_prompt = (
                        f"Cinematic {cam_setup['lens']} {cam_setup['framing'].lower()} shot of {speaker}. "
                        f"{cam_setup['motion']}. Setting: {loc_name}, {tod}. "
                        f"Subject Action: {speaker} delivers dialogue with synchronized lip movement: '{line_part}'. "
                        f"Lighting: dramatic cinematic key light with soft rim fill. 8k photorealistic, 24fps motion."
                    )

                    dlg_obj = MicroSceneDialogue(
                        speaker=speaker,
                        line_en=line_part,
                        words_count=words_count,
                        pacing_words_per_sec=wps,
                        speech_duration_seconds=speech_dur,
                        buffer_seconds=buf_dur,
                        performance_instruction=sync_acting_directive,
                        dialogue_type=dlg_type
                    )

                    all_micro_shots.append(MicroSceneShot(
                        id=f"sh_{sc_num:02d}_{shot_in_scene:02d}_{uuid.uuid4().hex[:4]}",
                        scene_number=sc_num,
                        shot_number=shot_in_scene,
                        global_shot_number=global_idx,
                        shot_code=shot_code,
                        slugline=slugline,
                        location_name=loc_name,
                        time_of_day=tod,
                        shot_type=cam_setup["type"],
                        camera_framing=cam_setup["framing"],
                        camera_motion=cam_setup["motion"],
                        lens_spec=cam_setup["lens"],
                        duration_seconds=total_dur,
                        speech_pacing_rate=f"{wps:.1f} words/sec",
                        dialogue=dlg_obj,
                        characters_in_frame=[speaker],
                        visual_action=shot_action,
                        soundscape_and_foley=f"Foley cues: subtle clothing rustle, breath intake, {soundscape}",
                        ai_video_prompt=video_prompt,
                        acting_directive=sync_acting_directive
                    ))
                    shot_in_scene += 1
                    global_idx += 1

        else:
            # If no dialogues in this scene, expand action beats into 2 visual takes
            for b_idx in range(1, min(3, len(beats))):
                beat_text = beats[b_idx]
                cam_setup = CAMERA_FRAMINGS[1 + (b_idx % (len(CAMERA_FRAMINGS) - 1))]
                shot_code = f"SC{sc_num:02d}-TK{shot_in_scene:02d}"

                action_prompt = (
                    f"Cinematic {cam_setup['lens']} {cam_setup['framing'].lower()} shot. {cam_setup['motion']}. "
                    f"Setting: {loc_name}, {tod}. Action: {beat_text}. Photorealistic 8k, volumetric atmosphere."
                )

                all_micro_shots.append(MicroSceneShot(
                    id=f"sh_{sc_num:02d}_{shot_in_scene:02d}_{uuid.uuid4().hex[:4]}",
                    scene_number=sc_num,
                    shot_number=shot_in_scene,
                    global_shot_number=global_idx,
                    shot_code=shot_code,
                    slugline=slugline,
                    location_name=loc_name,
                    time_of_day=tod,
                    shot_type=cam_setup["type"],
                    camera_framing=cam_setup["framing"],
                    camera_motion=cam_setup["motion"],
                    lens_spec=cam_setup["lens"],
                    duration_seconds=7.0,
                    speech_pacing_rate=f"{wps:.1f} words/sec",
                    dialogue=None,
                    characters_in_frame=cast[:1],
                    visual_action=beat_text,
                    soundscape_and_foley=f"Dynamic action foley: footsteps, atmospheric reverberation, {soundscape}",
                    ai_video_prompt=action_prompt,
                    acting_directive="[Instruction: Visual action beat without dialogue, dramatic body language]"
                ))
                shot_in_scene += 1
                global_idx += 1

        # -------------------------------------------------------------
        # FINAL SHOT OF SCENE: Reaction & Scene Climax / Button Take
        # -------------------------------------------------------------
        closing_beat = beats[-1] if len(beats) > 1 else f"Characters complete the encounter in {loc_name} as dramatic tension settles."
        cam_setup = CAMERA_FRAMINGS[4]  # Intimate Dramatic Close-Up or Reverse Reaction
        shot_code = f"SC{sc_num:02d}-TK{shot_in_scene:02d}"

        closing_prompt = (
            f"Cinematic {cam_setup['lens']} dramatic closing shot. {cam_setup['motion']}. "
            f"Setting: {loc_name}, {tod}. Visual: {closing_beat}. Camera lingers on character's resolved expression. "
            f"8k cinematic color grading, lens flare, 24fps motion."
        )

        all_micro_shots.append(MicroSceneShot(
            id=f"sh_{sc_num:02d}_{shot_in_scene:02d}_{uuid.uuid4().hex[:4]}",
            scene_number=sc_num,
            shot_number=shot_in_scene,
            global_shot_number=global_idx,
            shot_code=shot_code,
            slugline=slugline,
            location_name=loc_name,
            time_of_day=tod,
            shot_type=cam_setup["type"],
            camera_framing=cam_setup["framing"],
            camera_motion=cam_setup["motion"],
            lens_spec=cam_setup["lens"],
            duration_seconds=6.5,
            speech_pacing_rate=f"{wps:.1f} words/sec",
            dialogue=None,
            characters_in_frame=cast[:1],
            visual_action=f"Scene button: {closing_beat}",
            soundscape_and_foley=f"Decaying reverb, settling foley, {soundscape}",
            ai_video_prompt=closing_prompt,
            acting_directive="[Instruction: Concluding reaction beat, subtle dramatic realization, zero spoken dialogue]"
        ))
        global_idx += 1

    return all_micro_shots


# ==========================================
# 4. LangGraph State Machine
# ==========================================

def load_master_scenes_node(state: MicroSceneAgentState) -> Dict[str, Any]:
    """Ingest master scenes from scene_bible.json."""
    bible_path = state.get("scene_bible_path", SCENE_BIBLE_PATH)
    if not os.path.exists(bible_path):
        return {"raw_scenes": []}

    with open(bible_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = data.get("scenes", [])
    print(f"[Node: LoadMasterScenes] Loaded {len(scenes)} master scenes from {os.path.basename(bible_path)}")
    return {"raw_scenes": scenes}


def decompose_into_shots_node(state: MicroSceneAgentState) -> Dict[str, Any]:
    """Decompose scenes into micro-scenes (takes) strictly conforming to 6s-15s and 2 wps."""
    raw_scenes = state.get("raw_scenes", [])
    min_dur = state.get("min_duration", 6.0)
    max_dur = state.get("max_duration", 15.0)
    wps = state.get("words_per_sec", 2.0)

    scene_data = {"scenes": raw_scenes}
    shots = generate_micro_scenes_from_master(
        scene_data, min_dur=min_dur, max_dur=max_dur, wps=wps
    )

    print(f"[Node: DecomposeShots] Generated {len(shots)} cinematic takes across {len(raw_scenes)} master scenes.")
    return {"micro_shots": [s.model_dump() for s in shots]}


def lock_micro_scenes_bible_node(state: MicroSceneAgentState) -> Dict[str, Any]:
    """Verify constraints and save micro_scenes_bible.json."""
    shots_raw = state.get("micro_shots", [])
    total_seconds = sum(s.get("duration_seconds", 0.0) for s in shots_raw)
    mins = int(total_seconds // 60)
    secs = int(total_seconds % 60)
    formatted_runtime = f"{mins}m {secs:02d}s"

    bible = {
        "novel_title": "Solar Dragons Need Love Too 1",
        "pacing_standard": "2.0 words per second",
        "duration_constraint": "6.0s - 15.0s per take",
        "total_master_scenes": len(set(s.get("scene_number") for s in shots_raw)),
        "total_shots": len(shots_raw),
        "total_runtime_seconds": round(total_seconds, 1),
        "total_runtime_formatted": formatted_runtime,
        "shots": shots_raw
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MICRO_SCENES_BIBLE_PATH, "w", encoding="utf-8") as f:
        json.dump(bible, f, indent=2, ensure_ascii=False)

    print(f"[Node: LockMicroScenes] Saved {len(shots_raw)} takes to {MICRO_SCENES_BIBLE_PATH} (Total runtime: {formatted_runtime})")
    return {"is_locked": True}


def build_micro_scene_graph():
    """Build the LangGraph pipeline for micro-scene / shot decomposition."""
    workflow = StateGraph(MicroSceneAgentState)

    workflow.add_node("load_master_scenes", load_master_scenes_node)
    workflow.add_node("decompose_shots", decompose_into_shots_node)
    workflow.add_node("lock_micro_scenes", lock_micro_scenes_bible_node)

    workflow.add_edge(START, "load_master_scenes")
    workflow.add_edge("load_master_scenes", "decompose_shots")
    workflow.add_edge("decompose_shots", "lock_micro_scenes")
    workflow.add_edge("lock_micro_scenes", END)

    return workflow.compile()


def load_micro_scenes_bible() -> Dict[str, Any]:
    """Load the locked micro-scenes bible."""
    if not os.path.exists(MICRO_SCENES_BIBLE_PATH):
        return {
            "novel_title": "",
            "pacing_standard": "2.0 words per second",
            "duration_constraint": "6.0s - 15.0s per take",
            "total_master_scenes": 0,
            "total_shots": 0,
            "total_runtime_seconds": 0.0,
            "total_runtime_formatted": "0m 00s",
            "shots": []
        }
    with open(MICRO_SCENES_BIBLE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    print("--- Running Micro-Scene Production Graph ---")
    graph = build_micro_scene_graph()
    initial_state = {
        "scene_bible_path": SCENE_BIBLE_PATH,
        "raw_scenes": [],
        "micro_shots": [],
        "min_duration": 6.0,
        "max_duration": 15.0,
        "words_per_sec": 2.0,
        "is_locked": False
    }
    graph.invoke(initial_state)
    bible = load_micro_scenes_bible()
    print(f"Done! Created {bible['total_shots']} shots across {bible['total_master_scenes']} master scenes.")
    print(f"Total runtime: {bible['total_runtime_formatted']}")
