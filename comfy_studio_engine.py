"""
ComfyUI Multi-Model Studio Engine
Direct automation runner for the Studio's 4 core workflows:
1. Voice & Line Generation: Qwen3-TTS Voice Design (Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
2. Environment Matte Plates: Krea 2 Turbo (krea2_turbo_int8_convrot.safetensors)
3. Actor in Canonical Costume: Krea 2 Identity-Edit (krea2_identity_edit_v1_2.safetensors)
4. AI Video Generation: MiniMax H3 (Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors)
"""

import os
import sys
import json
import time
import shutil
import random
import requests
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOWS_DIR = os.path.join(BASE_DIR, "workflows")
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_DIR = r"C:\ComfyUI_windows_portable\ComfyUI"
COMFY_INPUT_DIR = os.path.join(COMFY_DIR, "input")
COMFY_OUTPUT_DIR = os.path.join(COMFY_DIR, "output")

os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
os.makedirs(COMFY_OUTPUT_DIR, exist_ok=True)

# Workflow File Paths
WF_VOICE_PATH = os.path.join(WORKFLOWS_DIR, "workflow_qwen3_voice_design.json")
WF_ENV_PATH = os.path.join(WORKFLOWS_DIR, "workflow_krea2_environment.json")
WF_ACTOR_OUTFIT_PATH = os.path.join(WORKFLOWS_DIR, "workflow_krea2_actor_outfit.json")
WF_PORTRAIT_PATH = os.path.join(WORKFLOWS_DIR, "workflow_krea2_portrait.json")
WF_VIDEO_PATH = os.path.join(WORKFLOWS_DIR, "workflow_minimax_h3_video.json")


def is_comfy_online() -> bool:
    """Check if ComfyUI instance is responding."""
    try:
        r = requests.get(f"{COMFY_URL}/system_stats", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def submit_comfy_prompt(workflow_dict: Dict[str, Any], client_id: str = "cinegraph_studio") -> str:
    """Send prompt graph to ComfyUI queue and return prompt_id."""
    res = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow_dict, "client_id": client_id}, timeout=10)
    if res.status_code == 200:
        return res.json().get("prompt_id", "")
    raise RuntimeError(f"ComfyUI submission error ({res.status_code}): {res.text}")


def poll_comfy_result(prompt_id: str, timeout_sec: int = 300) -> Dict[str, Any]:
    """Poll ComfyUI history until prompt completes or times out."""
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        try:
            h_res = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=5)
            if h_res.status_code == 200:
                h_data = h_res.json()
                if prompt_id in h_data:
                    entry = h_data[prompt_id]
                    status = entry.get("status", {})
                    if status.get("completed", False):
                        return {"status": "success", "outputs": entry.get("outputs", {})}
                    if status.get("status_str") == "error":
                        messages = status.get("messages", [])
                        return {"status": "error", "error": str(messages)}
        except Exception:
            pass
        time.sleep(1.5)
    return {"status": "timeout", "error": f"Execution timed out after {timeout_sec} seconds"}


def clean_slug(name: str) -> str:
    """Sanitize string for filename use."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_")


# ==========================================
# 1. Voice & Dialogue Generation (Qwen3-TTS)
# ==========================================
def generate_voice_qwen(
    character_name: str,
    line_en: str,
    voice_description: str,
    shot_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Renders spoken dialogue into .wav/.mp3 using Qwen3-TTS Voice Design.
    """
    if not os.path.exists(WF_VOICE_PATH):
        raise FileNotFoundError(f"Voice workflow missing: {WF_VOICE_PATH}")

    with open(WF_VOICE_PATH, "r", encoding="utf-8") as f:
        wf = json.load(f)

    slug = clean_slug(character_name)
    prefix_code = clean_slug(shot_code) if shot_code else "line"
    out_prefix = f"audio/cine_{slug}_{prefix_code}"

    # Node 1: Qwen3TTSVoiceDesign
    if "1" in wf:
        wf["1"]["inputs"]["text"] = line_en
        wf["1"]["inputs"]["voice_description"] = voice_description
        wf["1"]["inputs"]["seed"] = random.randint(100000000, 999999999)

    # Node 3: SaveAudioAdvanced
    if "3" in wf:
        wf["3"]["inputs"]["filename_prefix"] = out_prefix

    prompt_id = submit_comfy_prompt(wf, client_id=f"voice_{slug}")
    result = poll_comfy_result(prompt_id, timeout_sec=120)

    audio_files = []
    if result.get("status") == "success":
        outputs = result.get("outputs", {})
        # Node 3 outputs
        if "3" in outputs and "audio" in outputs["3"]:
            for a in outputs["3"]["audio"]:
                fname = a.get("filename", "")
                subfolder = a.get("subfolder", "audio")
                audio_files.append(f"/audio/{fname}" if not subfolder else f"/{subfolder}/{fname}")

    return {
        "status": result.get("status"),
        "prompt_id": prompt_id,
        "character": character_name,
        "audio_files": audio_files,
        "primary_audio": audio_files[0] if audio_files else None
    }


# ==========================================
# 2. Environment Matte Plate (Krea 2 Turbo)
# ==========================================
def generate_environment_krea(
    location_name: str,
    prompt_text: str,
    aspect_ratio: str = "16:9 (Widescreen)"
) -> Dict[str, Any]:
    """
    Renders 16:9 cinematic matte plate environment using Krea 2 Turbo.
    """
    if not os.path.exists(WF_ENV_PATH):
        raise FileNotFoundError(f"Environment workflow missing: {WF_ENV_PATH}")

    with open(WF_ENV_PATH, "r", encoding="utf-8") as f:
        wf = json.load(f)

    slug = clean_slug(location_name)
    out_prefix = f"environments/cine_{slug}"

    # Node 53: CR Text (Prompt input)
    if "53" in wf:
        wf["53"]["inputs"]["text"] = prompt_text

    # Node 49: ResolutionSelector
    if "49" in wf:
        wf["49"]["inputs"]["aspect_ratio"] = aspect_ratio

    # Node 30:3: KSampler (Seed)
    if "30:3" in wf:
        wf["30:3"]["inputs"]["seed"] = random.randint(100000000, 999999999)

    # Node 29: SaveImage
    if "29" in wf:
        wf["29"]["inputs"]["filename_prefix"] = out_prefix

    prompt_id = submit_comfy_prompt(wf, client_id=f"env_{slug}")
    result = poll_comfy_result(prompt_id, timeout_sec=120)

    image_files = []
    if result.get("status") == "success":
        outputs = result.get("outputs", {})
        if "29" in outputs and "images" in outputs["29"]:
            for img in outputs["29"]["images"]:
                fname = img.get("filename", "")
                subfolder = img.get("subfolder", "environments")
                image_files.append(f"/environments/{fname}" if not subfolder else f"/{subfolder}/{fname}")

    return {
        "status": result.get("status"),
        "prompt_id": prompt_id,
        "location": location_name,
        "image_files": image_files,
        "primary_image": image_files[0] if image_files else None
    }


# ==========================================
# 3. Actor in Costume (Krea 2 Identity-Edit)
# ==========================================
def generate_actor_outfit_krea(
    character_name: str,
    portrait_filename: str,
    outfit_prompt: str,
    aspect_ratio: str = "9:16 (Portrait Widescreen)"
) -> Dict[str, Any]:
    """
    Renders character in canonical wardrobe with locked facial identity using Krea 2 Edit.
    """
    if not os.path.exists(WF_ACTOR_OUTFIT_PATH):
        raise FileNotFoundError(f"Actor outfit workflow missing: {WF_ACTOR_OUTFIT_PATH}")

    with open(WF_ACTOR_OUTFIT_PATH, "r", encoding="utf-8") as f:
        wf = json.load(f)

    slug = clean_slug(character_name)
    out_prefix = f"character_outfits/cine_{slug}"

    # Ensure source portrait is in ComfyUI input folder
    src_portrait = os.path.join(COMFY_OUTPUT_DIR, "character_portraits", portrait_filename)
    dst_portrait = os.path.join(COMFY_INPUT_DIR, portrait_filename)
    if os.path.exists(src_portrait) and not os.path.exists(dst_portrait):
        shutil.copy2(src_portrait, dst_portrait)

    # Node 72: LoadImage (Source face portrait)
    if "72" in wf:
        wf["72"]["inputs"]["image"] = portrait_filename

    # Node 111: CR Text (Costume & action prompt)
    if "111" in wf:
        wf["111"]["inputs"]["text"] = outfit_prompt

    # Node 83: ResolutionSelector
    if "83" in wf:
        wf["83"]["inputs"]["aspect_ratio"] = aspect_ratio

    # Node 53: KSampler (Seed)
    if "53" in wf:
        wf["53"]["inputs"]["seed"] = random.randint(100000000, 999999999)

    # Node 29: SaveImage
    if "29" in wf:
        wf["29"]["inputs"]["filename_prefix"] = out_prefix

    prompt_id = submit_comfy_prompt(wf, client_id=f"actor_{slug}")
    result = poll_comfy_result(prompt_id, timeout_sec=120)

    image_files = []
    if result.get("status") == "success":
        outputs = result.get("outputs", {})
        if "29" in outputs and "images" in outputs["29"]:
            for img in outputs["29"]["images"]:
                fname = img.get("filename", "")
                subfolder = img.get("subfolder", "character_outfits")
                image_files.append(f"/character_outfits/{fname}" if not subfolder else f"/{subfolder}/{fname}")

    return {
        "status": result.get("status"),
        "prompt_id": prompt_id,
        "character": character_name,
        "image_files": image_files,
        "primary_image": image_files[0] if image_files else None
    }


# ==========================================
# 4. Final Video Generation (MiniMax H3 Ref2VA)
# ==========================================
def generate_video_shot_minimax(
    shot_code: str,
    prompt_6part: str,
    duration_seconds: float = 8.0,
    actor_image_filename: Optional[str] = None,
    env_image_filename: Optional[str] = None,
    audio_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Renders 6s-15s synchronized cinematic video clip using MiniMax H3 Ref2VA with TaoMate 3-Step LoRA.
    """
    if not os.path.exists(WF_VIDEO_PATH):
        raise FileNotFoundError(f"Video workflow missing: {WF_VIDEO_PATH}")

    with open(WF_VIDEO_PATH, "r", encoding="utf-8") as f:
        wf = json.load(f)

    slug = clean_slug(shot_code)
    out_prefix = f"video/cine_{slug}"
    clamped_dur = max(6.0, min(15.0, float(duration_seconds)))

    # Node 2: PrimitiveStringMultiline (The 6-part MiniMax multimodal prompt)
    if "2" in wf:
        wf["2"]["inputs"]["value"] = prompt_6part

    # Node 17: PrimitiveFloat (Duration in seconds)
    if "17" in wf:
        wf["17"]["inputs"]["value"] = clamped_dur

    # Node 129: RandomNoise (Seed)
    if "129" in wf:
        wf["129"]["inputs"]["noise_seed"] = random.randint(100000000, 999999999)

    # Node 221: Actor Image Reference (<Picture 1>)
    if actor_image_filename and "221" in wf:
        wf["221"]["inputs"]["image"] = actor_image_filename

    # Node 212: Environment Image Reference (<Picture 2>)
    if env_image_filename and "212" in wf:
        wf["212"]["inputs"]["image"] = env_image_filename

    # Node 222: Spoken Audio Reference (<Audio 1>)
    if audio_filename and "222" in wf:
        wf["222"]["inputs"]["audio"] = audio_filename

    # Node 135 & 229: VHS_VideoCombine output prefixes
    if "135" in wf:
        wf["135"]["inputs"]["filename_prefix"] = out_prefix
    if "229" in wf:
        wf["229"]["inputs"]["filename_prefix"] = f"{out_prefix}_rtx"

    prompt_id = submit_comfy_prompt(wf, client_id=f"vid_{slug}")
    result = poll_comfy_result(prompt_id, timeout_sec=600)  # Video rendering can take longer

    video_files = []
    if result.get("status") == "success":
        outputs = result.get("outputs", {})
        for node_id in ("229", "135"):
            if node_id in outputs and "gifs" in outputs[node_id]:
                for v in outputs[node_id]["gifs"]:
                    fname = v.get("filename", "")
                    subfolder = v.get("subfolder", "video")
                    video_files.append(f"/video/{fname}" if not subfolder else f"/{subfolder}/{fname}")

    return {
        "status": result.get("status"),
        "prompt_id": prompt_id,
        "shot_code": shot_code,
        "duration_seconds": clamped_dur,
        "video_files": video_files,
        "primary_video": video_files[0] if video_files else None
    }
